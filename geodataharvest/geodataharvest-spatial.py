import geodata_harvester as gh

lat, lon = -35.274603, 149.098498
buffer = 0.05
left, top, right, bottom = lon - buffer, lat - buffer, lon + buffer, lat + buffer 
[left, top, right, bottom]

df = gh.harvest.run("parameters.yaml", preview=True, return_df=True)


