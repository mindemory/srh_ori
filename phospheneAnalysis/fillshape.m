function XY = fillshape(polyshape_common, screenYpixels)
% computes all pixels within bounds defined by polyshape_common
% Fist, the minimum and maximum x-coords of the
% polyshape_common are computed. Using an iterative procedure, column
% vectors from min_x to max_x spanning whole screen are created.
% isinterior() is used to determine which of these pixels are inside
% polyshape_common.
min_x = round(min(polyshape_common.Vertices(:, 1)));
max_x = round(max(polyshape_common.Vertices(:, 1)));
XY = NaN(1, 2);
for xx = min_x:max_x
    row_pnts = [repmat(xx, [screenYpixels 1]), (1:screenYpixels)'];
    TFin = isinterior(polyshape_common, row_pnts);
    XY = [XY; row_pnts(TFin, :)];
end
XY = XY(2:end, :);
end