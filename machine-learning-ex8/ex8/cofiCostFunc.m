function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%



predict_movie_rating = X * Theta';

movie_rating_error =  Y - predict_movie_rating;

error_factor = R .* movie_rating_error ;

theta_reg = (lambda/2) * sum(sum(Theta.^2))

x_reg = (lambda/2) * sum(sum(X .^2));

J = (1/2) * sum(sum(error_factor.^2)) + theta_reg + x_reg;
%
%X_grad = R .*(movie_rating_error - Y) * Theta + (lambda * X);
% 
%Theta_grad = (R .*(movie_rating_error - Y))' * X + (lambda * Theta);


% calculating gradient of x.
for i = 1:num_movies
  idx = find(R(i,:) == 1);  % users that have rated movie i.
  Theta_temp = Theta(idx, :); % user features of movie i.
  Y_temp  = Y(i, idx);        % user's ratings of movie 
  X_grad(i,:) = (X(i,:) * Theta_temp' - Y_temp) * Theta_temp + lambda * X(i, :);
end

% calculating gradient of theta.
for i = 1:num_users    
  idx = find(R(:, i) == 1);   % movies that have rated by user i.
  X_temp = X(idx,:);           % features of movies rated by user j
  Y_temp = Y(idx,i);            % user ratings by user j.
  Theta_grad(i,:) = (X_temp * Theta(i, :)' - Y_temp)' * X_temp + lambda * Theta(i, :);
end
%   =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
