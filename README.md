# Optical Flow Estimation Comparision

## Patch-based Lucas-Kanade 
Compute spatially local gradient in patches, and then compute LeastSquare solution between gradient of t-1, t  
It's based on $$I(x,y,t) = I( x +dx, y + dy, t+dt) \\ I_x*v I_y*u = -I_t$$
![LK_SingleLayer 결과](output/LK_SingleLayer.gif)  

## Feature-based GaussNewton-SingleLayer  
Feature detector : ORB 
![GaussNewton SingleLayer 결과](output/GaussNewton_SingleLayer.gif)  

## Feature-based GaussNewton-MultiLayer  
Feature detector : ORB
Coarse-to-Fine optical flow estimation using image pyramid  
<!--https://github.com/SeungPyo-Jeon/OpticalFlow-Comparision/blob/main/output/GaussNewton_MultiLayer?raw=true-->
![GaussNewton_MultiLayer 결과](output/GaussNewton_MultiLayer.gif)  
