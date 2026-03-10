# Optical Flow Estimation Comparision

## Patch-based Lucas-Kanade 
![LK_SingleLayer 결과](output/LK_SingleLayer.gif)  

## Feature-based GaussNewton-SingleLayer  
Feature detector : ORB / GFTT  
![GaussNewton SingleLayer 결과](output/GaussNewton_SingleLayer.gif)  

## Feature-based GaussNewton-MultiLayer  
Feature detector : ORB / GFTT  
Coarse-to-Fine optical flow estimation using image pyramid  
<!--https://github.com/SeungPyo-Jeon/OpticalFlow-Comparision/blob/main/output/GaussNewton_MultiLayer?raw=true-->
![GaussNewton_MultiLayer 결과](output/GaussNewton_MultiLayer.gif)  
