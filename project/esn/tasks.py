
from celery import Celery
from project.esn.fitter.logger import logger

app = Celery('test_randomMatrix')
app.config_from_object('project.esn.celeryconf')

@app.task
def test_0(kwargs):
    import project.esn.test as t
    logger.debug(f"{kwargs}")
    mse = t.Test(**kwargs)()
    return {"args": kwargs,
            "mse":mse}
