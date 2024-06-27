%function [pixCoords] = va2pixel(phi)
function [pix1, pix2] = va2pixel(r, theta)
% created by Mrugank (06/15/2022):
% rho is distance of screen from the subject, and angle is the visual angle
% of the stimulus
global parameters screen;
%%%%%%%%%
% % ro and angle are in degree on the screen
% X = (ro * cosd(angle))/Sc.pixWidth;
% Y = (ro * sind(angle))/Sc.pixWidth;
% 
% pixCoords(1,1) = round(X) + Sc.xCenter;
% pixCoords(1,2) = round(Y) + Sc.yCenter;
%%%%%%
rho = parameters.viewingDistance;
r_cm = rho * tand(phi/2);
pix1 = round(r_cm/screen.pixWidth); % convert cm to pixels
pix2 = round(r_cm/screen.pixHeight); % convert cm to pixels
rho = parameters.viewingDistance;
ecc1_cm = rho * tand(phi);
ecc2_cm = rho * (tand(phi + theta) - tand(phi));
pix1 = round(ecc1_cm/screen.pixWidth);
pix2 = round(ecc2_cm/screen.pixWidth);
end



