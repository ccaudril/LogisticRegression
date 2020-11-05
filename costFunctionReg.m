function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% COSTFUNCTION

sum1=0;
for i=1:m
  sum1 = sum1 - y(i)*log(sigmoid(X(i,:)*theta)) - (1-y(i))*log(1-sigmoid(X(i,:)*theta));
endfor

sum2=0;
for j=2:size(theta)
  sum2 += theta(j)*theta(j);
endfor

J = (sum1/m) + ((sum2*lambda)/(2*m));


% GRADIENT

% for j=0
sum3=0;
for i=1:size(X,1)
  sum3 += (sigmoid(X(i,:)*theta)-y(i))*X(i,1);
endfor
grad(1) += sum3/m;

% for j>=1
for j = 2:size(theta)
  sum4=0;
  for i=1:size(X,1)
    sum4 += (sigmoid(X(i,:)*theta)-y(i))*X(i,j);
  endfor
  grad(j) += sum4/m+theta(j)*lambda/m;
endfor


% =============================================================

end
