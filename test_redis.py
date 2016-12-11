import redis
from redis_queue import RedisQueue

q = RedisQueue('flappy')

for i in range(120):
    q.put('item' + str(i))
