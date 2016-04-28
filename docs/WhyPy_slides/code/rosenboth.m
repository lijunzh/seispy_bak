function [f, g] = rosenboth(x)
% Calculate objective f
f = sum(100.0*(x(2:end)-x(1:end-1).^2.0).^2.0 + (1-x(1:end-1)).^2.0);

if nargout > 1 % gradient required
    xm = x(2:end-1);
    xm_m1 = x(1:end-2);
    xm_p1 = x(3:end);
    g = zeros(size(x));
    g(2:end-1) = 200.*(xm-xm_m1.^2) - 400.*(xm_p1 - xm.^2).*xm - 2.*(1-xm);
    g(1) = -400.*x(1)*(x(2)-x(1).^2) - 2*(1-x(1));
    g(end) = 200.*(x(end)-x(end-1)).^2;
end