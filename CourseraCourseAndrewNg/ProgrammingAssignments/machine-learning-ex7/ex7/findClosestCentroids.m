function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

m = size(X,1);

for i = 1:m 
  % initialize a distance vector for this x(i) of size 1 x K
  % Each element in this vector will be distance between x(i) and centroid mu(k)
  distanceXi = zeros(1, K);
  
  for k = 1:K
    % compute || x(i) - mu(k) || and find the k with lowest of that value
    % for any two vectors a and b, || a - b|| = norm(a - b, 2)
    distanceXi(1,k) = norm(X(i,:) - centroids(k,:), 2);
  end
  
  % inner loop is done; find the min distance and its index. set it to ci
  [minDistance, minDistanceIndex] = min(distanceXi);
  idx(i) = minDistanceIndex;
end

% ====== without using min function =====
% for i = 1:m
  % min_k = 1;
  
  % initialize min_norm to the norm of first two x(i) and mu(k)
  % min_norm = norm(X(i,:) - centroids(1,:), 2);
  % for k = 1:K
    % compute || x(i) - mu(k) || and find the k with lowest of that value
    % for any two vectors a and b, || a - b|| = norm(a - b, 2)
    % currentNorm = norm(X(i,:) - centroids(k,:), 2);
    % if ( currentNorm <= min_norm)
      % min_norm = currentNorm;
      % min_k = k;
    % endif
  % end
  
  % inner loop is done; we will have the min k by now. set it to ci
    % idx(i) = min_k;
% end




% =============================================================

end

