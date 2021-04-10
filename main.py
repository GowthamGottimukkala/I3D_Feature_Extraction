from pathlib import Path
import shutil
import argparse
import subprocess
import sys
import numpy as np
import os
import time
from extraction.extract_features import run

def generate(datasetpath, outputpath, load_model_rgb, load_model_flow, sample_mode, frequency, batch_size):
	tempdir = outputpath+ "/temp/"
	rootdir = Path(datasetpath)
	videos = [str(f) for f in rootdir.glob('**/*.mp4')]
	for video in videos:
		startime = time.time()
		Path(tempdir).mkdir(parents=True, exist_ok=True)
		print("Generating for {0}".format(video))
		process = subprocess.run('build/denseFlow_gpu --vidFile={0} --outFolder={1}'.format(video, tempdir), shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True)
		if(process.returncode == 0):
			print("halfdone")
			savepath = outputpath + "/" + video.split("/")[-1].split(".")[0] + "/"
			Path(savepath).mkdir(parents=True, exist_ok=True)
			rgb = run("rgb", load_model_rgb, sample_mode, frequency, tempdir, batch_size)
			flow = run("flow", load_model_flow, sample_mode, frequency, tempdir, batch_size)
			np.save(savepath + "feature_rgb", rgb)
			np.save(savepath + "feature_flow", flow)
		shutil.rmtree(tempdir)
		print("done in {0}.".format(time.time() - startime))

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str)
	parser.add_argument('--outputpath', type=str)
	parser.add_argument('--load_model_rgb', type=str, default="models/rgb_imagenet.pt")
	parser.add_argument('--load_model_flow', type=str, default="models/flow_imagenet.pt")
	parser.add_argument('--sample_mode', type=str, default="center_crop")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=1)
	args = parser.parse_args()
	generate(args.datasetpath, str(args.outputpath), args.load_model_rgb, args.load_model_flow, args.sample_mode, args.frequency, args.batch_size)    
