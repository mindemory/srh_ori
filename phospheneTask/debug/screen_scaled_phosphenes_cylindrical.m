%% Mrugank
% This code was written to scale the phosphene maps from an old screen to a
% new screen. The cylindrical version accounts for the change in the eye
% distance from the screen.
clear all; clc;

subjID = '01'; session = '02';
%temp_tmsRtnTpy = load('Results/sub27/tmsRtnTpy_sub27_sess03.mat');
temp_tmsRtnTpy = load('Results/backup/sub27/tmsRtnTpy_sub27_sess01.mat');

loadDIR = ['Results/sub' subjID];

saveDIR = ['Results/sub' subjID];
data = [loadDIR '/tmsRtnTpy_sub' subjID '_sess' session];
load(data);
xx_old = tmsRtnTpy.Params.screen.xCenter
yy_old = tmsRtnTpy.Params.screen.yCenter
zz_old = tmsRtnTpy.Params.screen.viewDist

tmsRtnTpy.Params.screen = temp_tmsRtnTpy.tmsRtnTpy.Params.screen;
xx_new = tmsRtnTpy.Params.screen.xCenter
yy_new = tmsRtnTpy.Params.screen.yCenter
zz_new = tmsRtnTpy.Params.screen.viewDist

cc = tmsRtnTpy.Response.Drawing.coords;
for ii = 1:length(cc)
    if ~isnan(cc{1, ii})
        dely = (cc{1, ii}(:, 2) - yy_old);
        delx = (cc{1, ii}(:, 1) - xx_old);
        delz = zz_old;
        
        rho = sqrt(delx.^2 + dely.^2 + delz.^2);
        theta = atan2(-dely, (delx + 0.00001));
        phi = acos(delz./(rho + 0.00001));
        
        cc_x = xx_new + rho .* sin(phi) .* cos(theta);
        cc_y = yy_new - rho .* sin(phi) .* sin(theta);
        cc_z = zz_new;
        if mean(zz_new - rho .* cos(phi)) >= 0.0001
            disp('issues')
        end
%        cc_x = xx_new - xx_old - cc{1, ii}(:, 1);
%        cc_y = yy_new + yy_old - cc{1, ii}(:, 2);
        
        cc{1, ii} = [cc_x, cc_y];
    end
end

tmsRtnTpy.Response.Drawing.coords = cc;
saveName = [saveDIR '/tmsRtnTpy_sub' subjID '_sess' session];
save(saveName,'tmsRtnTpy')

calcPhospheneArea(subjID, session, 1);
calcStimLocations(subjID,session)
