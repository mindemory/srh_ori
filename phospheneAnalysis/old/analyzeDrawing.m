function [area, border] = analyzeDrawing(drawing, parameters)
% The part for calculating the area coordinates needs to be re-written.
% Right now, some columns are missed in the calculated areas. 
drawing = round(drawing); % In case pixels are not already rounded.
% scrH_pix = tmsRtnTpy.Params.screen.screenYpixels;
% scrW_pix = tmsRtnTpy.Params.screen.screenXpixels;
drawing(drawing <= 0) = 1;

%% throw away redundant coordinates
drawing_unique = unique(drawing,'rows','stable');
% I1 = zeros(scrH_pix,scrW_pix);
% for i = 1:length(drawing)
%     disp(i);
%     I1(drawing(i,2),drawing(i,1)) = 1;
% end
% 
% figure(); imagesc(I1);

%% find the closest point to each point
k = 1;
drawing_tmp = drawing_unique;
while size(drawing_tmp, 1) > 1
    A = drawing_tmp(1, :);
    drawing_tmp(1, :) = [];
    B = drawing_tmp;
    distances = sqrt(sum((B - A).^2, 2));
    % find the smallest distance and use that as an index into B:
    closest_tmp = B(distances == min(distances), :);
    closest(k, :) = closest_tmp(1, :);
    k = k + 1;
end

%% connect the each point to it's closest point
border = [];
for i = 1:size(closest,1)
    xy1 = drawing_unique(i,:);
    xy2 = closest(i,:);
    xyConnected = connectPoints(xy1, xy2);
    border = [border; xyConnected];
end

border = connect_missing_points(border);

% throw away redundant coordinates
border = unique(border,'rows','stable');
% I1 = zeros(scrH_pix,scrW_pix);
% for i = 1:length(drwng_connected)
%     I1(drwng_connected(i,2),drwng_connected(i,1)) = 1;
% end
% figure(); imagesc(I1);

%% fill in the shape
area = [];
for x = 1:parameters.screenXpixels
    indsY = find(border(:,1) == x);
    if length(indsY) > 1
        y1 = min(border(indsY,2));
        y2 = max(border(indsY,2));
        inds_y = y1:y2;
        area = [area ; [repmat(x,[length(inds_y) 1 ]) inds_y'] ];
    end
end
% throw away redundant coordinates
area = unique(area,'rows','stable');

% I1 = zeros(scrH_pix,scrW_pix);
% for i = 1:length(inds_area)
%     I1(inds_area(i,2),inds_area(i,1)) = 1;
% end
% figure(); imagesc(I1);

