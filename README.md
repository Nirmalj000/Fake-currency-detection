# Fake-currency-detection
An android App which replies you you with the currency is original or fake from the picture you captured.
In this proposed system, a fake note detection using various method. In image pre-processing the image was cropped, adjusted and smoothed. Then the image converted into 
gray scale. 
After conversion the edges are detected. Next the image segmentation is applied. 
After segmentation the features are extracted. Finally compared and find the currency original or fake. The complete methodology works for Indian denomination 10, 20, 50, 100, 200, 500 and 2000. The method is very simple and easy to implement. This technique is very adaptive to implement in real time world.
![image](https://user-images.githubusercontent.com/54202985/180791116-0eff8957-42be-4852-aae9-fcacfa3c67b7.png)
## Methodology
• The user has to open the app to capture an image.                          
• This image is uploaded to the real - time database AWS server in order to obtain real-time results.   
• The image is then fed to the CNN model and the produced results will be displayed on the 
screen within fraction of seconds.   
• Flask will be working as backend in order to compute the image.  
• The image is uploaded to the real time database so that the real time result can be computed.   
• Image which is in the database is fed to the CNN model and the predicted results are pushed 
back into the database.   
• Conversion of the image to 224 x 224 pixels and image pre-processing is performed.   
• After the results have been predicted by the model and uploaded to the database, the android 
application fetches the results instantly and the results will be displayed on the app through 
JSON object.  
## Results
![image](https://user-images.githubusercontent.com/54202985/180795164-1cf04556-647b-4229-a4a4-a44d925fbdb5.png)
![image](https://user-images.githubusercontent.com/54202985/180795261-8a296f82-7156-4229-8fa0-ec37e4f95dd0.png)
