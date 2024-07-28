function drawTexturesTest(screen, targX, targY, stimSize, cpd, contrast, gaussWidth)

if nargin < 7
    gaussOn = 0;
else
    gaussOn = 1;
end
ori_target = 45;
% targX = tMap.targX(trial);
% targY = tMap.targY(trial);

[x_grid, y_grid] = meshgrid(-stimSize/2:stimSize/2, -stimSize/2:stimSize/2);
gratingMat =  screen.grey * contrast * cos(2*pi*cpd/stimSize*y_grid+0);
% Add a circular mask
[c, r] = meshgrid(1:stimSize+1, 1:stimSize+1);
circMask = sqrt((c-stimSize/2).^2 + (r-stimSize/2).^2) <= stimSize/2;

if gaussOn == 1
    % Add a gaussian mask
    [x, y] = meshgrid(linspace(-stimSize/2, stimSize/2, stimSize+1));
    gaussMask = exp(-(x.^2 + y.^2) / (2 * gaussWidth^2));
    combinedMask = circMask .* gaussMask;
    gratingMat = screen.grey + gratingMat .* combinedMask;
else
    gratingMat = screen.grey + gratingMat .* circMask;
end

gratingTex = Screen('MakeTexture', screen.win, gratingMat);
Screen('DrawTexture', screen.win, gratingTex, [], ...
    CenterRectOnPoint([0 0 stimSize stimSize], targX, targY), 360-ori_target); 

end
