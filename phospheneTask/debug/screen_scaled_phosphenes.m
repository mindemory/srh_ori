%% Mrugank
% This code was written to scale the phosphene maps from an old screen to a
% new screen. The non-cylindrical version fails to account for the change in the eye
% distance from the screen.
clear all; clc;

subjID = '01'; session = '01';
%temp_tmsRtnTpy = load('Results/sub27/tmsRtnTpy_sub27_sess03.mat');
temp_tmsRtnTpy = load('Results/backup/sub27/tmsRtnTpy_sub27_sess03.mat');

loadDIR = ['Results/sub' subjID];

saveDIR = ['Results/sub' subjID];
data = [loadDIR '/tmsRtnTpy_sub' subjID '_sess' session];
load(data);
xx_old = tmsRtnTpy.Params.screen.xCenter
yy_old = tmsRtnTpy.Params.screen.yCenter

tmsRtnTpy.Params.screen = temp_tmsRtnTpy.tmsRtnTpy.Params.screen;
xx_new = tmsRtnTpy.Params.screen.xCenter
yy_new = tmsRtnTpy.Params.screen.yCenter

cc = tmsRtnTpy.Response.Drawing.coords;
for ii = 1:length(cc)
    if ~isnan(cc{1, ii})
        dely = (cc{1, ii}(:, 2) - yy_old);
        delx = (cc{1, ii}(:, 1) - xx_old);
        
        r = sqrt(dely.^2 + delx.^2);
        theta = atan2(-dely, (delx + 0.00001));
        
        cc_x = xx_new + r .* cos(theta);
        cc_y = yy_new - r .* sin(theta);
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
