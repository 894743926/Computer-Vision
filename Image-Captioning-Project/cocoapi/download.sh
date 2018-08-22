mkdir "annotations" && mkdir "images"
mkdir "images/train2014"
mkdir "images/val2014"
mkdir "images/test2014"

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
unzip -o  "train2014.zip" -d "images/train2014"
unzip -o  "test2014.zip" -d "images/test2014"
unzip -o  "val2014.zip" -d "images/val2014"
unzip -o  "annotations_trainval2014.zip" -d "annotations"
unzip -o  "image_info_test2014.zip" -d "annotations"


echo "All done,now remove zip files"
rm -rf *.zip

echo "All set..."
