from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from ray import serve


# 1: Define a FastAPI app and wrap it in a deployment with a route handler.
app = FastAPI()


@serve.deployment(name="QAModel")
class QAModel:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tknzr = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    
    @serve.batch(
        max_batch_size=64,
        batch_wait_timeout_s=0.1,
    )
    async def handle_batch(self, question, context):
        conditioned_doc = "<P> " + " <P> ".join([d for d in context[0]])
        query_and_docs = "question: {} context: {}".format(question[0], conditioned_doc)

        model_input = self.tknzr(query_and_docs, truncation=True, padding=True, return_tensors="pt")

        generated_answers_encoded = self.model.generate(input_ids=model_input["input_ids"].to(self.device),
                                            attention_mask=model_input["attention_mask"].to(self.device),
                                            min_length=64,
                                            max_length=256,
                                            do_sample=False, 
                                            early_stopping=True,
                                            num_beams=8,
                                            temperature=1.0,
                                            top_k=None,
                                            top_p=None,
                                            eos_token_id=self.tknzr.eos_token_id,
                                            no_repeat_ngram_size=3,
                                            num_return_sequences=1)
        return self.tknzr.batch_decode(generated_answers_encoded, skip_special_tokens=True,clean_up_tokenization_spaces=True)


@serve.deployment(name="ZSClassifier")
class ZSClassifier:
    def __init__(self, model_name):
        self.model = pipeline("zero-shot-classification", model=model_name)
    
    @serve.batch(
        max_batch_size=64,
        batch_wait_timeout_s=0.1,
    )
    async def handle_batch(self, text, candidates, multi_label=False):
        return [self.model(text[0], candidates[0], multi_label=multi_label)]


@serve.deployment(route_prefix="/", name="REWrapper",)
@serve.ingress(app)
class REWrapper:
    def __init__(self, qa_model, zs_classifier, questions, positive_candidates, negative_candidates):
        self.qa_model = qa_model
        self.zs_classifier = zs_classifier
        self.questions = questions
        self.positive_candidates = positive_candidates
        self.negative_candidates = negative_candidates
        self.candidate_labels = positive_candidates + negative_candidates
    
    def process_candidates(self, zh_res, positive_candidates, negative_candidates) -> list[float]:
        positive_prob, negative_prob = 0, 0
        for label, score in zip(zh_res["labels"], zh_res["scores"]):
            if label in positive_candidates:
                positive_prob += score
            elif label in negative_candidates:
                negative_prob += score
        return [
            positive_prob/len(positive_candidates), 
            negative_prob/len(negative_candidates),
            ]

    async def classify_answer(self, text, ner_pair) -> list[list[float]]:
        # create candidates for sh
        candidates = []
        for c in range(len(self.candidate_labels)):
            candidates.append(self.candidate_labels[c].format(ner_pair[0], ner_pair[1]))
        # make predictions per question
        probs_per_question = []
        # iterate over questions
        for question in self.questions:
            # create QA input and get QA model's output
            qa_res = await self.qa_model.handle_batch.remote(
                question.format(ner_pair[0], ner_pair[1]),
                text,
            )
            qa_res = await qa_res
            # get classification results
            zh_res = await self.zs_classifier.handle_batch.remote(qa_res, candidates, multi_label=False)
            zh_res = await zh_res
            # 
            y_pred = self.process_candidates(
                zh_res,
                [c.format(ner_pair[0], ner_pair[1]) for c in self.positive_candidates], 
                [c.format(ner_pair[0], ner_pair[1]) for c in self.negative_candidates], 
                )
            probs_per_question.append(y_pred)
        return [probs_per_question, qa_res]
        
    @app.post("/re")
    async def handle_request(self, text: list[str], ner_pair: tuple[str, str]):
        y_preds, answer = await self.classify_answer(text, ner_pair)
        positive = 0.0
        negative = 0.0
        for q in y_preds:
            positive += q[0]
            negative += q[1]
        return {
            "probability of positive": positive/len(y_preds),
            "probability of negative": negative/len(y_preds),
            "explanation": answer,
        }


qa_model = QAModel.bind("vblagoje/bart_lfqa")
zs_clsfr = ZSClassifier.bind("facebook/bart-large-mnli")
re_wrapper = REWrapper.bind(
    qa_model,
    zs_clsfr,
    [
        "Is the {} involved in the development or progression of {}?",
        "Does the {} have a known association with the {}?",
        "Are there any studies that suggest a connection between the {} and the {}?"
    ],
    [
        "{} is strongly implicated in the development or progression of {}",
        "{} has a moderate association with the {}",
    ],
    [
        "The relationship between {} and {} is uncertain or unclear",
        "{} has no known connection to the {}",
    ],
)
