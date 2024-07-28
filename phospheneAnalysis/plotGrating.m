function plotGrating(parameters, targX, targY, stimSize, cpd, contrast, gaussWidth)

    if nargin < 4
        stimSize = parameters.stimSize; % in degrees
        % Compute stimSize in pixels
        stimSize = tand(stimSize) * parameters.viewingDistance / parameters.pixSize;
    
    end
    
    if nargin < 5
        cpd = 7;
    end
    
    if nargin < 6
        contrast = 1;
    end
    
    if nargin < 7
        gaussWidth = 0.65 * (stimSize/4);
    end
    
    gaussOn = 1;
    % targY = resolution(1) - targY;
    ori_target = 45;
    background = 128;
    % targX = tMap.targX(trial);
    % targY = tMap.targY(trial);
    
    [x_grid, y_grid] = meshgrid(-stimSize/2:stimSize/2, -stimSize/2:stimSize/2);
    gratingMat =  background * contrast * cos(2*pi*cpd/stimSize*y_grid+0);
    % Rotate the grating 
    gratingMat = imrotate(gratingMat, ori_target, 'bilinear', 'crop');
    % Add a circular mask
    [c, r] = meshgrid(1:stimSize+1, 1:stimSize+1);
    circMask = sqrt((c-stimSize/2).^2 + (r-stimSize/2).^2) <= stimSize/2;
    
    if gaussOn == 1
        % Add a gaussian mask
        [x, y] = meshgrid(linspace(-stimSize/2, stimSize/2, stimSize+1));
        gaussMask = exp(-(x.^2 + y.^2) / (2 * gaussWidth^2));
        combinedMask = circMask .* gaussMask;
        gratingMat = background + gratingMat .* combinedMask;
    else
        gratingMat = background + gratingMat .* circMask;
    end
    
    % Embed the grating in a larger matrix
    fullMat = 128 * ones([parameters.screenXpixels, parameters.screenYpixels]);
    % res = min(parameters.screenXpixels, parameters.screenYpixels);
    % fullMat = 128 * ones([res, res]);
    % fullMat(targY-stimSize/2:targY+stimSize/2, targX-stimSize/2:targX+stimSize/2) = gratingMat;
    fullMat(targX-stimSize/2:targX+stimSize/2, targY-stimSize/2:targY+stimSize/2) = gratingMat;
    
    % Draw the full matrix
    imshow(fullMat', [0, 255]);
    end
    