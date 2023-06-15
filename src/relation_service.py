import time


class RelationService:
    def __init__(self, conn):
        self.conn = conn
    
    def update_data(self, query, rows, batch_size = 10000):
        total = 0
        batch = 0
        start = time.time()
        result = None

        while batch * batch_size < len(rows):

            res = self.conn.query(query, parameters={'rows': rows[batch*batch_size:(batch+1)*batch_size].to_dict('records')})
            total += res[0]['total']
            batch += 1
            result = {"total":total, "batches":batch, "time":time.time()-start}
            print(result)

        return result
    

    def add_ners_rels(self, rows, batch_size=10000):
        query = '''
        //chunk1 NERs
        UNWIND $rows as row
        MERGE(n1:NER{name:row.chunk1}) ON CREATE SET n1.type=row.entity1
        //chunk2 NERs
        MERGE(n2:NER{name:row.chunk2}) ON MATCH SET n2.type=row.entity2

        //connect NERs
        WITH row, n1, n2
        MERGE (n1)-[:LINKS{relation:row.relation}]->(n2)

        WITH n1
        MATCH (n1)
        RETURN count(*) as total  
        '''

        return self.update_data(query, rows, batch_size)
