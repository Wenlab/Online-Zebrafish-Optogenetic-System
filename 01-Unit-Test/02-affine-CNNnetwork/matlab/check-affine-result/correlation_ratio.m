% This is a function to calcluate CR (correlation ratio)
% between 2 images X and Y
% the formula of CR can be looked up by the definition on wikipedia:
% https://en.wikipedia.org/wiki/Correlation_ratio

% the loop could be replaced with "accumarry" for better performance
% @Chen Shen; cshen@ustc.edu.cn

function eta = correlation_ratio(X, Y)
X = X(:); % convert 3D to column vector
Y = Y(:); % convert 3D to column vector
L = max(X);
mYx = zeros(1, L+1); % to write mean per class here
nx = zeros(1, L+1);  % to write number of samples per class here
for i = unique(X).'
   Yn = Y(X == i);
   if numel(Yn)>1
      mYx(i+1) = mean(Yn);
      nx(i+1) = numel(Yn);
   end
end
mY = mean(Y);        % mean across all samples
eta = sqrt(sum(nx .* (mYx - mY).^2) / sum((Y-mY).^2));