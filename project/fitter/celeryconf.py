broker_url='redis://192.168.178.31:6379/2'
#result_backend='redis://192.168.178.31:6379/3'
result_backend='mongodb://192.168.178.31:27017/from_celery'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'Europe/Amsterdam'
enable_utc = True
