MODEL=model23

if [ -f "$MODEL.t7" ];
then
	echo "Cannot generate model file, '$MODEL.t7' is already exist."
	exit
fi

th "$MODEL.lua"
