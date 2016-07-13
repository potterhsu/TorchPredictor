MODEL=model23

if [ -f "$MODEL.tpa" ];
then
	echo "Cannot generate model file, '$MODEL.tpa' is already exist."
	exit
fi

th ModelPrinter.lua "$MODEL.t7" > "$MODEL.tpa"
