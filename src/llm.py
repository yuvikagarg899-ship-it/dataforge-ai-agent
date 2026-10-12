import time
from google.genai import errors 
def generate_with_retry(client,attempts:int = 4,base_delay:float = 2.0,**kwargs):
    for i in range(attempts):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServeError:
            if i == attempts-1:
                raise
            time.sleep(base_delay*(2**i))