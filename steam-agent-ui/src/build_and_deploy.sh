npm run build
echo "Build Success"
sudo cp -r ./dist/* /var/www/html/
echo "Moved to /var/www/html/"