% this function performs linear regression using gradient descent. y
% is a column vector of outputs, X is a (N,p+1) matrix (N is the number of 
% data points, p the number of features). Each row of X contains an entry
% equal to 1 in the first column, and the inputs associated with the p 
%features in the remaining columns alpha is the learning rate
% for gradient descent. tol is the tolerance that defines convergence of
% gradient descent.
 
 
function [beta, iteration] = myLinearRegression(y,X,alpha,tol)
N=size(X,1);% number of samples
p=size(X,2);%number of parameters (it includes the intercept)
beta=zeros(p,1);%parameters
err=1.0;
iteration=1;
while (err>tol)
    J=(X*beta-y)'*(X*beta-y);%objective function (RSS)
    gradient = 2*((X*beta-y)'*X)';% gradient of J
    beta = beta -alpha/N*gradient;% taking a step along the gradient
    err=abs(J-(X*beta-y)'*(X*beta-y));%computing the change in J
    iteration=iteration+1;
end

myLinearRegression(600,80000,0.54,1)

