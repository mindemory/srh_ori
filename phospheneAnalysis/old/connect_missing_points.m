function drawing_connected = connect_missing_points(drawing_connected)
    % if any points are missing between two consecutive points in
    % drawing_connected, that is to say, if there is a gap greater than 1
    % pixel unit between two consecutive coordinates, this fixes by drawing
    % lines parallel to x-axis between the two consecutive points. A simple
    % approximation to create a closed figure.
    x_size = size(drawing_connected, 1);
    ii = 1;
    while ii < x_size
        if abs(drawing_connected(ii, 1) - drawing_connected(ii+1, 1)) <= 1
            ii = ii + 1;
        elseif drawing_connected(ii, 1) < drawing_connected(ii+1, 1)
            X = drawing_connected(ii, 1):drawing_connected(ii+1, 1);
            lenX = length(X);
            Y = drawing_connected(ii, 2) * ones(1, lenX);
            XY = [X; Y]';
            x_size = x_size + lenX;
            drawing_connected = [drawing_connected(1:ii, :); XY; drawing_connected(ii+1:end, :)];
            ii = ii + lenX - 1;
        elseif drawing_connected(ii, 1) > drawing_connected(ii+1, 1)
            X = drawing_connected(ii, 1):-1:drawing_connected(ii+1, 1);
            lenX = length(X);
            Y = drawing_connected(ii, 2) * ones(1, lenX);
            XY = [X; Y]';
            x_size = x_size + lenX;
            drawing_connected = [drawing_connected(1:ii, :); XY; drawing_connected(ii+1:end, :)];
            ii = ii + lenX - 1;
        end
    end
end