function points = errorToPoints(error)

% if error < 5
%     points = 100 - (20 * error /5);
% else
%     points = 100 * (1 - (error/180));
% end

% lambda = 0.005;
% max_points = 100;
% points = max_points * exp(-lambda * error);
% points = max(0, points);

alpha = 2.7; beta = 50;
max_points = 100;
max_err = 90;
y = gampdf(error, alpha, beta);
y_max = gampdf(max_err, alpha, beta);
points = max_points * (1 - y / y_max);
points = max(0, points);
end
