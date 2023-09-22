function p = get_steady_state(w)
% dt = 1e-3;
% m = size(w, 1);
% p = rand(m, 1);
% p = p/sum(p);
% while any(w*p>1e-12)
%     p = p + w*p*dt;
% end
[V,D] = eig(w);
% steady-state has real eigenvector
V = real(V);
% find eigenvalue 1
[~, entry] = min(abs(diag(D) - 1));
% normalize eigenvector
p = V(:,entry)/sum(V(:,entry));

end