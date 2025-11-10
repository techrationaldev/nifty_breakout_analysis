#To START/STOP/RESTART the project service on server use this commands ->

sudo systemctl daemon-reload
sudo systemctl enable breakoutdetector.service
sudo systemctl start breakoutdetector.service
sudo systemctl status breakoutdetector.service
sudo systemctl restart breakoutdetector.service
