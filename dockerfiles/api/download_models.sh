# download netvlad
wget https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat
mkdir -p /app/api/third_party/netvlad
mv Pitts30K_struct.mat /app/api/third_party/netvlad/VGG16-NetVLAD-Pitts30K.mat
# dowload loftr
wget https://raw.githubusercontent.com/MACILLAS/LoFTR/master/weights/outdoor_ds.ckpt
mkdir -p /app/api/third_party/LoFTR/weights
mv outdoor_ds.ckpt /app/api/third_party/LoFTR/weights/outdoor_ds.ckpt
