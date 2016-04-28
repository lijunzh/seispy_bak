data = randn(100);
% Normalize data
data = bsxfun(@rdivide,data,std(data,[],1));
% alternatively
data = data ./ repmat(std(data,[],1),size(data,1),1);





fun = @(x) sum(100.0*(x(2:end)-x(1:end-1).^2.0).^2.0 ...
    + (1-x(1:end-1)).^2.0);
x0 = [1.3, 0.7, 0.8, 1.9, 1.2];
options = optimoptions(@fminunc,'Display','iter',...
    'Algorithm','quasi-newton');
[x,fval,exitflag,output] = fminunc(fun,x0,options);




A = randn(3); b = randn(3,1)
x = A \ b
