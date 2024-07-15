
from concurrent import futures
import logging

import grpc

from protogen import TableDetection_pb2
from protogen import TableDetection_pb2_grpc

from TableDetectorService import TableDetectorService


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    TableDetection_pb2_grpc.add_TableDetectionServiceServicer_to_server(TableDetectorService(), server)
    server.add_insecure_port('[::]:50051')
    
    print('starting server in port 50051')
    server.start()
    print('started server in port 50051')
    server.wait_for_termination()



if __name__ == '__main__':
    logging.basicConfig()
    serve()
