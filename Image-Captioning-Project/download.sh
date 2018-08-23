pip install -r requirements.txt

mkdir "cocoapi/annotations" && mkdir "cocoapi/images"
mkdir "cocoapi/images/train2014"
mkdir "cocoapi/images/val2014"
mkdir "cocoapi/images/test2014"

echo "download train images"
wget "http://images.cocodataset.org/zips/train2014.zip"
echo "download test images"
wget "http://images.cocodataset.org/zips/test2014.zip"
echo "download val images"
wget "http://images.cocodataset.org/zips/val2014.zip"

echo "download annotations"
wget "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
echo "download test annotations"
wget "http://images.cocodataset.org/annotations/image_info_test2014.zip"

echo "Unzip all the downloaded zip files"
unzip -o  "train2014.zip" -d "cocoapi/images/train2014"
unzip -o  "test2014.zip" -d "cocoapi/images/test2014"
unzip -o  "val2014.zip" -d "cocoapi/images/val2014"
unzip -o  "annotations_trainval2014.zip" -d "cocoapi/annotations"
unzip -o  "image_info_test2014.zip" -d "cocoapi/annotations"


echo "All done,now remove zip files"
rm -rf *.zip

echo "All set..."
