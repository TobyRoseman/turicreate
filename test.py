import turicreate as tc; tc.config.set_num_gpus(0)

data = tc.load_sframe('/Volumes/Data/hapt_data.sframe/')
m = tc.load_model('../../../ac.model/')

m.predict(data)
