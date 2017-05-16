from tasks import add 
from celery import group

r=group(add.s(i, i) for i in range(2))()
print(r.get())
