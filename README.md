## To compile
pyinstaller --windowed build.spec --add-data 'model1.h5:.'  --add-data 'vgg16.h5:.' 
python3 -m PyInstaller -F --windowed ./build.spec --add-data 'model1.h5:.'  --add-data 'vgg16.h5:.' 