function r_pix = va2pixel(parameters, screen, phi)
% created by Mrugank (06/15/2022):
% rho is distance of screen from the subject, and angle is the visual angle
% of the stimulus
rho = parameters.viewingDistance;
r_cm = rho * tand(phi/2);
r_pix = round(r_cm/screen.pixSize); % convert cm to pixels
end