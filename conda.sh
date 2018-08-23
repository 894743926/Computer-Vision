echo "Which system, A: MacOS, B:Linux"
read -p "Choose: " system
echo $system

if [ $system = 'A' ];
then
    echo You have a mac
    url="https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
elif [ $system = 'B' ];
then
    echo You have a linux
    url="https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
else
    echo Wrong answer
fi

echo "Download miniconda..."

echo $url

curl "$url" > "conda.sh"

bash conda.sh

echo "remove the conda installer"

rm -rf conda.sh


