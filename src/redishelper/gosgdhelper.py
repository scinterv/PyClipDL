import pickle
import codecs
import redis
from redis import ConnectionPool, Redis

class GoSGDHelper(object):
    def __init__(self,host='localhost',port=6379,db=0,cluster=False):
        self.pool = ConnectionPool(host=host,port=port,decode_responses=True)
        self.r = Redis(connection_pool=self.pool)

        # GoSGD Redis keys
        self._key_prefix="GoSGD:"
        self.edge_id = self._key_prefix + "EdgeID"
        self.edge_set = self._key_prefix + "EdgeIDSet"
        self.time_cost = self._key_prefix + "TimeCost"

        self.ID = 1
        self.NAME = self._key_prefix + "edge:" + str(self.ID)

        self.push_ok = self._key_prefix + "pushNum"
        self.update_ok = self._key_prefix + "updateNum"

        """prune_flag
        -1: initial state alse to be pruned
        0: to be pruned
        1: prune finished
        """
        self.prune_flag = self._key_prefix + "prune"

    # sign a new edge, assign a new edge ID and add the id to edge id set
    def signin(self):
        try:
            lua = """
                local eid = redis.call('incr','{0}')
                redis.call('sadd','{1}',eid)
                redis.call('setnx','{2}',-1)
                return eid
            """.format(self.edge_id,self.edge_set,self.prune_flag)
            self.ID = self.r.eval(lua,0)
            self.NAME = self._key_prefix + "edge:" + str(self.ID)
        except Exception as e:
            print(repr(e))
            print("Sign in a new edge failed!")

    # return the number of current edge id set
    def cur_edge_num(self):
        num = 0
        try:
            num = self.r.scard(self.edge_set)
        except Exception as e:
            print(repr(e))
            print("Get number of edge failed!")
        return num

    # insert epoch time cost of current edge, then increase the number of finish pushing
    def ins_time(self,epoch_time):
        try:
            pipe = self.r.pipeline()
            pipe.zadd(self.time_cost,{self.NAME: epochtime})
            pipe.incr(self.push_ok,amount=1)
            pipe.execute()
        except Exception as e:
            print(repr(e))
            print("Insert epoch time cost of {} failed!".format(self.NAME))

    # insert epoch time cost and model parameters of current edge, then increase the number of finish pushing
    def ins_time_params(self,other_eid,epoch_time,score,params):
        try:
            push_key = self._key_prefix + "edge:" + str(other_eid)
            pickled = self.p2str((score,params))

            pipe = self.r.pipeline(transaction=True)
            pipe.zadd(self.time_cost,{self.NAME: epoch_time})
            pipe.lpush(push_key,pickled)
            pipe.incr(self.push_ok,amount=1)
            pipe.execute()
        except Exception as e:
            print(repr(e))
            print("Insert epoch time cost and model params of {} failed!".format(self.NAME))

    # get the min and max epoch  time of all edges, then increase the number of finish updateing
    def min2max_time(self):
        try:
            pipe = self.r.pipeline(transaction=True)
            pipe.zrange(self.time_cost,0,-1,desc=False,withscores=True)
            pipe.incr(self.update_ok,amount=1)

            time_list = pipe.execute()[0]
        except Exception as e:
            print(repr(e))
            print("Get min and max epoch time cost from {} failed!".format(self.NAME))
        return time_list[0][1], time_list[-1][1]

    # get the min and max epoch tiem also model parameters of all edges, then increase the number of finish updateing
    def min2max_time_params(self):
        params_list = []
        try:
            pipe = self.r.pipeline(transaction=True)
            pipe.zrange(self.time_cost,0,-1,desc=False,withscores=True)
            pipe.lrange(self.NAME,0,-1)
            pipe.incr(self.update_ok,amount=1)
            full_res = pipe.execute()

            time_list = full_res[0]
            pickled_list = full_res[1]
            params_list = list(map(self.str2p,pickled_list))
        except Exception as e:
            print(repr(e))
            print("Get min and max epoch time cost and params from {} failed!".format(self.NAME))
        return time_list[0][1], time_list[-1][1], params_list

    # judge if all edges finish pushing epoch time cost or parameters
    def finish_push(self):
        try:
            lua = """
            local edgenum = redis.call('scard','{0}')
            local push = redis.call('get','{1}')
            local flag = 0
            if tonumber(push) == edgenum then
                redis.call('set','{2}',0)
                flag = 1
            end
            return flag
            """.format(self.edge_set,self.push_ok,self.prune_flag)
            res = self.r.eval(lua,0)
        except Exception as e:
            print(repr(e))
            print("Judge finish push failed on {}!".format(self.NAME))
        return int(res) == 1

    # judge if all edges finish updating epoch time cost or parameters
    def finish_update(self):
        try:
            lua = """
            local prune = redis.call('get','{0}')
            if prune == '1' then
                return 1
            end
            local edgenum = redis.call('scard','{1}')
            local update = redis.call('get','{2}')
            local flag = 0
            if tonumber(update) == edgenum then
                redis.call('set','{3}',1)
                redis.call('del','{4}')
                redis.call('del','{5}')
                redis.call('del','{6}')
                for _,k in ipairs(redis.call('keys','{7}edge:*')) do redis.call('del',k) end
                flag = 1
            end
            return flag
            """.format(self.prune_flag,self.edge_set,self.update_ok,self.prune_flag,
            self.time_cost,self.push_ok,self.update_ok,self._key_prefix)

            res = self.r.eval(lua,0)
        except Exception as e:
            print(repr(e))
            print("Judge finish update failed on {}!".format(self.NAME))
        return int(res) == 1

    # register out an edge and remove related redis keys
    def register_out(self):
        try:
            lua = """
            local edgenum = redis.call('scard','{0}')
            if edgenum == 1 then
                for _,k in ipairs(redis.call('keys','{1}*')) do redis.call('del',k) end
            else
                redis.call('srem','{2}',{3})
                redis.call('del','{4}')
                redis.call('decr','{5}')
            end
            """.format(self.edge_set,self._key_prefix,self.edge_set,self.ID,self.NAME,self.push_ok)
            self.r.eval(lua,0)
        except Exception as e:
            print(repr(e))
            print("register out failed {}!".format(self.NAME))

    # return a random edge id from edge id set
    def random_edge_id(self,can_be_self=False):
        selid = self.ID
        try:
            selid = int(self.r.srandmember(self.edge_set))
            #print(selid)
            while can_be_self == False and selid == self.ID:
                selid = int(self.r.srandmember(self.edge_set))
        except Exception as e:
            print(repr(e))
            print("Choose rand edge id failed on {}".format(self.NAME))
        return selid

    """
        Pickle and unpickle to portable string
    """
    def p2str(self, obj):
        return codecs.encode(pickle.dumps(obj), "base64").decode()

    def str2p(self, pickled):
        return pickle.loads(codecs.decode(pickled.encode(), "base64"))
