echo "MAIN SCRIPT"
echo "===================="
pylint ./main.py

echo "GPU INFO"
echo "===================="
pylint ./menus/gpu_info

echo "LOCAL IMAGE GENERATION"
echo "===================="
pylint ./menus/local_image_generation

echo "LOCAL TEXT GENERATION"
echo "===================="
pylint ./menus/local_text_generation

echo "ONLINE IMAGE GENERATION"
echo "===================="
pylint ./menus/online_image_generation

echo "ONLINE TEXT GENERATION"
echo "===================="
pylint ./menus/online_text_generation

echo "GENERAL MODULE"
echo "===================="
pylint ./modules/general