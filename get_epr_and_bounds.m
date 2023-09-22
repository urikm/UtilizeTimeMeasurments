function [epr, s_KLD, s_aff, passive] = get_epr_and_bounds(w)
% epr and passive
lamda = sum(w, 1);
m = size(w, 1);
w(1:m+1:end) = -lamda;
p = get_steady_state(w);

passive = (w(1, 2) * p(2) - w(2, 1) * p(1)) * log(w(1, 2) * p(2) / (w(2, 1) * p(1)));
epr = get_epr(w, p);

% % informed
% w_st = w;
% w_st(1, 1) = w_st(1, 1) + w_st(2, 1);
% w_st(2, 1) = 0;
% w_st(2, 2) = w_st(2, 2) + w_st(1, 2);
% w_st(1, 2) = 0;
% p_st = get_steady_state(w_st);
% 
% informed = (w(1, 2) * p(2) - w(2, 1) * p(1)) * log(w(1, 2) * p_st(2) / (w(2, 1) * p_st(1)));

[s_aff, s_KLD] = get_kld();

function [aff, kld] = get_kld()
    s = sym('s');
    t = sym('t');
    Psi_H = sym('Psi_H');
    N = length(w);
    omega = -diag(w);
   
    p12=-w(1,2)/w(2,2); pH22=1-p12;
    p21=-w(2,1)/w(1,1); pH11=1-p21;
   
   
    pH1=squeeze(w(3:end,1)./omega(1)); pH1=pH1/sum(pH1);
    pH2=squeeze(w(3:end,2)./omega(2)); pH2=pH2/sum(pH2);
   
    for i=1:N-2
        for j=1:N-2
            Psi_H(i,j) = w(i+2,j+2)/(omega(j+2)+s);
            if i==j
                Psi_H(i,j) = 0;
            end
        end
    end

    a=inv(diag(ones(1,N-2))-Psi_H);
   
    for i=1:N-2
        v1H(i) = w(1,i+2)/(omega(i+2)+s);
        v2H(i) = w(2,i+2)/(omega(i+2)+s);
    end
   
%     Psi_1H1=ilaplace(v1H*a*pH1);
    Psi_1H2 = ilaplace(v1H*a*pH2);
    Psi_2H1 = ilaplace(v2H*a*pH1);
%     Psi_2H2=ilaplace(v2H*a*pH2);
    
%     p1H1=double(int(Psi_1H1,t,0,Inf));
    p1H2 = double(subs(v1H*a*pH2, 0));
    p2H1 = double(subs(v2H*a*pH1, 0));
%     p2H2=double(int(Psi_2H2,t,0,Inf));
   
    Psi_2H1_normalized = Psi_2H1/p2H1;
    Psi_1H2_normalized = Psi_1H2/p1H2;
      
    R1=p12+p1H2*pH22;
    R2=p21+p2H1*pH11;
    RH1=pH11*R1;
    RH2=pH22*R2;
    sumR=R1+R2+RH1+RH2;

    R1=R1/sumR;     R2=R2/sumR;
    RH1 = RH1/sumR; RH2 = RH2/sumR;
    
    tau1 = -1/w(1,1);
    tau2 = -1/w(2,2);
    tauH1 = - double(subs(diff(v1H*a*pH1 + v2H*a*pH1, s),0));
%     disp(- double(subs(diff(v2H*a*pH1, s),0)))
    tauH2 = - double(subs(diff(v1H*a*pH2 + v2H*a*pH2 ,s),0));
%     disp(- double(subs(diff(v1H*a*pH2, s),0)))
    timePerStep = tau1*R1 + tau2*R2 + tauH1*RH1 + tauH2*RH2;

    js = R1*p21-R2*p12;
    d1 = js * log ( p21*p1H2*pH22 /(p12*p2H1*pH11) );
    integrand = (R1.*pH11.*p2H1.*Psi_2H1_normalized-R2.*pH22.*p1H2.*Psi_1H2_normalized).*log(Psi_2H1_normalized./Psi_1H2_normalized);
    d2 = double(int(integrand,t,0,Inf));
    kld = (d1+d2)/timePerStep;
    aff = d1/timePerStep;
end

function [epr_tmp] = get_epr(w, p)
    tol = 1e-15;
    n = transpose(w).*p;
    epr_tmp = 0;
    for i=1:m-1
        for j=i+1:m
            epr_tmp = epr_tmp + (n(i, j) - n(j, i))*log((n(i, j) + tol) / (n(j, i) + tol));
        end
    end
end
end