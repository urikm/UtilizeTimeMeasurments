close all; 
% Written by Gili Bisker
% bisker@tauex.tau.ac.il
% Analytical calculation of S_KLD

clc;  clear all;
set(0,'DefaultAxesFontSize',10); set(0,'defaultTextFontSize',10); set(0,'defaultAxesFontName', 'Calibri'); set(0,'defaultTextFontName', 'Calibri');
link=[1,2]; %choose link to calculate masked EP

syms s Psi_H t 
%% work with random matrix
N=4;
% W=round(100000*rand(N));
% W(1,3) = 0; W(3,1) =0;
% % load('W')
% W=[...
%     0  2   0  1;
%     3  0   2  35;
%     0  50  0  0.7;
%     8  0.2 75 0];
% 
% W=1000*W;


W = ones(N,N);
W(1,3) = 0; W(3,1) =0;
% W(4,3) = 0.1; W(3,4) =0.1;
W(1,2) = 0.01; W(2,1) =0.01;
% W(1,4) = 0.1; W(4,1) =0.1;

yy=20;
epss=0.5;
zz=yy/4;
kk=45;
W = [0,     10, zz, zz;
    kk,     0,  yy,  yy;
    kk/2,   5,  0,  epss;
    kk/2,   5,  epss,   0];



W=W-diag(sum(W));

W_stall=W; W_stall(1,2)=0; W_stall(2,1)=0; W_stall=W_stall-diag(sum(W_stall));
P_ss_st = find_ss(W_stall);
x_st=1/2*log(P_ss_st(1)/P_ss_st(2) * W(2,1)/W(1,2));

% x=linspace(-0.2,0.8,200);
% x=-0.2:0.005:0.8;
% x=-0.2:0.05:0.8;
% x=(((x_st)-3*abs(x_st)) : abs(x_st)/2 : ((x_st)+12*abs(x_st))) + 0.001;


%x=-2:0.5:2;
x=[0];sort([-2:0.5:2,x_st])
j_12=zeros(size(x));
length(x)

for i=1:length(x)
   i
   Wx=W;
    %Wx(1,2)= W(1,2)*exp(x(i));
%    Wx(2,1)= W(2,1)*exp(-x(i));

   %Wx(4,2)= W(4,2)*exp(x(i));
   %Wx(2,4)= W(2,4)*exp(x(i));
   
   Wx(4,3)= W(4,3)*exp(x(i));
   
   Wx=Wx-diag(sum(Wx));
   P_ss = find_ss(Wx);
  
   P_ss_track(i,:)=P_ss;
   j_12(i)=Wx(1,2)*P_ss(2)-Wx(2,1)*P_ss(1);
   Wx_stall=Wx; Wx_stall(1,2)=0; Wx_stall(2,1)=0;  Wx_stall=Wx_stall-diag(sum(Wx_stall));
   P_ss_st = find_ss(Wx_stall);
   ratio_ss_st(i)=P_ss_st(2)/P_ss_st(1);
   
%    P_ss_st_1(i)=P_ss_st(1)
%    P_ss_st_2(i)=P_ss_st(2)
   
   EP_link(i)=j_12(i) * log ( Wx(1,2)*P_ss(2) / Wx(2,1)/P_ss(1));
   EP_Matteo(i)=j_12(i) * log ( Wx(1,2)*P_ss_st(2) / Wx(2,1)/P_ss_st(1));
  
   EP_per_step(i)=calculate_entropy_production_per_step(Wx);
   [EP_Juan(i),affinity_Juan(i)]=Juan_Parrondo_Entropy(Wx,s,Psi_H,t);
   avg_step_time(i)=calculate_avg_step_time(Wx);
   
   EP_tot(i)=0;
   for k1=1:N
       for k2=1:N
           if (Wx(k1,k2)~=0 && Wx(k2,k1)~=0)
                EP_tot(i)=EP_tot(i)+ 1/2 * ( Wx(k1,k2)*P_ss(k2)-Wx(k2,k1)*P_ss(k1) ) * log ( Wx(k1,k2)*P_ss(k2) / Wx(k2,k1)/P_ss(k1) ); 
           end
       end
   end
   
end

% figure
% plot(x,EP_per_step,x,EP_Juan, x,EP_tot.*avg_step_time,'x',x,EP_Matteo.*avg_step_time); xlabel('x');
% legend('Totel EP per step','EP Juan per step','EP tot * step time','Matteo per step');

figure('Position',[300, 300,800,260]);
subplot(1,3,1)
plot(x,EP_link,x,EP_Matteo,'x-',x,EP_Juan,'o-', x,EP_tot); xlabel('F');
title('All the bounds')
legend('EP passive rate ','EP informed rate', 'EP KLD rate','EP tot rate','location','north');
subplot(1,3,2)
semilogy(x,EP_link,x,EP_Matteo,'x-',x,EP_Juan,'o-', x,EP_tot); xlabel('F'); ylim([1e-5 1e3])
title('All the bounds, log scale')
%legend('EP single link rate ','EP Matteo rate', 'EP Juan rate','EP tot rate');
subplot(1,3,3)
plot(x,EP_link,x,EP_Matteo,'x-',x,EP_Juan,'o-'); xlabel('F'); 
title('All the bounds, near stalling')
xlim([((x_st)-1) ((x_st)+1)])

figure('Position',[300, 600,1000,460]);
subplot(1,3,1)
plot(x,EP_link,x,EP_Matteo,x,affinity_Juan,'-.',x,EP_Juan,'--', x,EP_tot); xlabel('x');
legend('EP single link rate ','EP passive partial rate','EP Affinity', 'EP KLD rate','EP tot rate');
subplot(1,3,2)
plot(x,affinity_Juan-EP_Matteo)
title('Affinity minus Informed')
subplot(1,3,3)
plot(x,EP_tot-EP_Juan)
title('Total minus KLD')

min(EP_tot-EP_Juan)

% save numerical_EP_2017_09_12
% save x x
% save EP_link EP_link
% save EP_Matteo EP_Matteo
% save EP_Juan EP_Juan
% save x_st x_st
% save EP_tot EP_tot


% figure
% plot(x,EP_link,x,EP_Matteo,x,EP_tot,x,EP_CTRW); xlabel('x'); ylabel('j_{12}'); legend('Shiraishi','Matteo','Total','CTRW');
% xl=xlim; yl=ylim;
% figure
% plot(x,EP_link,x,EP_Matteo,x,EP_tot,x,ttruest); xlabel('x'); ylabel('j_{12}'); legend('Shiraishi','Matteo','Total','Parrondo');
% xlim(xl); ylim(yl);

% figure
% plot(x,j_12); xlabel('x'); ylabel('j_{12}');
% 
% figure
% plot(x,ratio_ss,x,ratio_ss_st,x,j_12,':k'); xlabel('x'); legend('p_2/p_1 (ss)','p_2/p_1 (stalling)','j_{12}');
% 
% figure
% plot(x,ratio_ss_st,x,P_ss_st_1,x,P_ss_st_2); xlabel('x'); legend('p_2/p_1 (ss)','p_1 (stalling)','p_2 (stalling)');
% 
% %% uni-cyclic network
% N=6;
% W=round(100*rand(N));
% W=diag(diag(W,1),1) + diag(diag(W,-1),-1) + diag(diag(W,N-1),N-1) + diag(diag(W,-(N-1)),-(N-1));
% W=W-diag(sum(W));
% 
% W_st=W; W_st(1,2)=0; W_st(2,1)=0; W_st=W_st-diag(sum(W_st));
% P_ss = find_ss(W);
% j_12=W(1,2)*P_ss(2)-W(2,1)*P_ss(1);
% P_ss_st = find_ss(W_st);
% ratio_ss_st=P_ss_st(2)/P_ss_st(1);
% 
% EP_link=j_12 * log ( W(1,2)*P_ss(2) / W(2,1)/P_ss(1))
% EP_Matteo=j_12 * log ( W(1,2)*P_ss_st(2) / W(2,1)/P_ss_st(1))
% EP_tot=0;
% for k1=1:N
%     for k2=1:N
%         if W(k1,k2)~=0
%             W(k1,k2)*P_ss(k2)-W(k2,k1)*P_ss(k1)
%         EP_tot=EP_tot+ 1/2 * [ W(k1,k2)*P_ss(k2)-W(k2,k1)*P_ss(k1) ] * log ( W(k1,k2)*P_ss(k2) / W(k2,k1)/P_ss(k1) );
%         end
%     end
% end
% EP_tot
% %%
% for i=1:length(x)
%     Wx=W;
%     Wx(1,2)= W(1,2)*exp(x(i));
%     Wx(2,1)= W(2,1)*exp(-x(i));
%     Wx=Wx-diag(sum(Wx));
%     P_ss = find_ss(Wx);
%     
%     
%     j_12(i)=Wx(1,2)*P_ss(2)-Wx(2,1)*P_ss(1);
%     Wx_stall=Wx; Wx_stall(1,2)=0; Wx_stall(2,1)=0;  Wx_stall=Wx_stall-diag(sum(Wx_stall));
%     P_ss_st = find_ss(Wx_stall);
%     
%     
%     for j=1:length(x)
%             
%     EP_link_mat(i,j)      =EP_link(i);
%     EP_Matteo_mat(i,j)    =EP_Matteo(i);
%     EP_arbitrary_mat(i,j) =j_12(i) * log ( Wx(1,2)*P_ss_track(j,2) / Wx(2,1)/P_ss_track(j,1));
%     EP_tot_mat(i,j)=EP_tot(i);
%     end
% end
% 
% 
% figure
% mesh(EP_tot_mat); hold on;
% mesh(EP_Matteo_mat)
% mesh(EP_arbitrary_mat)
% 
% %% change rates. keep delta_E and delta_mu constant
% r=logspace(-3,2,400);
% q=logspace(-3,2,200);
% r_prime=10;
% q_prime=5;
% delta_mu_prime=log(10);
% 
% delta_E=log(40)
% delta_mu=log(4)
% EP_link=0; EP_Matteo=0; EP_tot=0;
% 
% for r_ind=1:length(r)
%     for q_ind=1:length(q)
%         W=make_transition_matrix(r(r_ind),q(q_ind),delta_E,delta_mu,r_prime,q_prime,delta_mu_prime);
%         P_ss = find_ss(W);
%         
%         %W_st=find_stalling_matrix(r(r_ind),q(q_ind),delta_E,delta_mu,r_prime);
%         W_st=make_transition_matrix(0,q(q_ind),delta_E,delta_mu,r_prime,q_prime,delta_mu_prime); %same but with r=0
%         P_ss_st = find_ss(W_st);
%         
%         ratio(r_ind,q_ind) = P_ss_st(2)/P_ss_st(1)  - P_ss(2)/P_ss(1);
%         ratio_p_st(r_ind,q_ind)=P_ss_st(2)/P_ss_st(1);
%         
%         EP(r_ind,q_ind)=calculate_entropty_production_ss(W,P_ss);
%         EP_Shiraishi(r_ind,q_ind)=calculate_EP_Masked(W,P_ss,link);
%         EP_Matteo(r_ind,q_ind)=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
%         EP_total_minus_Matteo(r_ind,q_ind)=calculate_EP_total_minus_Matteo(W,P_ss,P_ss_st);
%         %check that the sum of Shiraishi's over all links is the total EP
%         if abs(check_calculation(W,P_ss,EP(r_ind,q_ind)))>1e-6; 
%             error('error'); 
%         end
%         diff(r_ind,q_ind)=EP_Matteo(r_ind,q_ind)-EP_Shiraishi(r_ind,q_ind);
%         if diff(r_ind,q_ind)<0
%             diff(r_ind,q_ind)
%         end
%         EP_Shiraishi_link(r_ind,q_ind,:)=calculate_Shiraishi_for_all_links(W,P_ss);
%         EP_Shiraishi_link_st(r_ind,q_ind,:)=calculate_Shiraishi_for_all_links(W_st,P_ss_st);
%     end
% end
% 
% [R,Q]=meshgrid(r,q);
% figure
% surf(R,Q,(EP-(EP_Matteo+EP_total_minus_Matteo))','EdgeColor','none'); set(gca,'XScale','log'); set(gca,'YScale','log'); xlabel('r');  ylabel('q');xlim([min(r),max(r)]); ylim([min(q),max(q)]);
% title('EP^{total} - (EP^{Matteo} + EP^{rest}) = 0');
% 
% figure
% surf(R,Q,ratio','EdgeColor','none'); set(gca,'XScale','log'); set(gca,'YScale','log'); xlabel('r');  ylabel('q');xlim([min(r),max(r)]); ylim([min(q),max(q)]);
% title('P_2/P_1 (Matteo - Shiraishi)');
% 
% 
% figure
% surf(R,Q,(EP-EP_Matteo)','EdgeColor','none'); set(gca,'XScale','log'); set(gca,'YScale','log'); xlabel('r');  ylabel('q');xlim([min(r),max(r)]); ylim([min(q),max(q)]);
% title('EP(Total)- EP(Matteo)');
% 
% figure
% surf(R,Q,ratio_p_st','EdgeColor','none'); 
% set(gca,'XScale','log'); set(gca,'YScale','log'); xlabel('r');  ylabel('q');xlim([min(r),max(r)]); ylim([min(q),max(q)]);
% title('P_2^{ST}/P_1^{ST}');
% 
% figure
% surf(R,Q,diff','EdgeColor','none'); set(gca,'XScale','log'); set(gca,'YScale','log'); xlabel('r');  ylabel('q');xlim([min(r),max(r)]); ylim([min(q),max(q)]);
% title('EP(Matteo) - EP(Shiraishi)');
% min(min(diff))
% 
% f=plot_results_as_function_of_rates(delta_E,delta_mu,r,q,EP);
% 
% %% change delta_E and delta_mu. keep rates constant
% r=1;
% q=1;
% EP=0;
% 
% delta_E=(linspace(-7,7,400));
% delta_mu=(linspace(-7,7,200));
% EP=0; 
% 
% for E_ind=1:length(delta_E)
%     for mu_ind=1:length(delta_mu)
%         W=make_transition_matrix(r,q,delta_E(E_ind),delta_mu(mu_ind),r_prime,q_prime,delta_mu_prime);
%         P_ss = find_ss(W);
%         
%         %W_st=find_stalling_matrix(r,q,delta_E(E_ind),delta_mu(mu_ind),r_prime);
%         W_st=make_transition_matrix(0,q,delta_E(E_ind),delta_mu(mu_ind),r_prime,q_prime,delta_mu_prime); %same but with r=0
%         P_ss_st = find_ss(W_st);
%         
% %         close all
% %         figure; plot(P_ss); hold on; plot(P_ss_st,':')
% %         title(['P_{ss}(2)/P_{ss}(1)=' num2str(P_ss(2)/P_ss(1))   '  P_{ss}^{ST}(2)/P_{ss}^{ST}(1)=' num2str(P_ss_st(2)/P_ss_st(1))])
% 
%         ratio(E_ind,mu_ind) = P_ss_st(2)/P_ss_st(1)  - P_ss(2)/P_ss(1);
%         ratio_p_st(E_ind,mu_ind)=P_ss_st(2)/P_ss_st(1) ;
%         
%         EP(E_ind,mu_ind)=calculate_entropty_production_ss(W,P_ss);  
%         EP_Shiraishi(E_ind,mu_ind)=calculate_EP_Masked(W,P_ss,link);
%         EP_Matteo(E_ind,mu_ind)=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
%         %check that the sum of Shiraishi's over all links is the total EP
%         if abs(check_calculation(W,P_ss,EP(E_ind,mu_ind)))>1e-6; 
%             %error('error'); 
%         end
%         diff(E_ind,mu_ind)=EP_Matteo(E_ind,mu_ind)-EP_Shiraishi(E_ind,mu_ind);
%         if diff(E_ind,mu_ind)<0
%             diff(E_ind,mu_ind)
%         end
%     end
% end
% 
% [E,MU]=meshgrid(delta_E,delta_mu);
% figure
% surf(E,MU,ratio','EdgeColor','none'); 
% %set(gca,'XScale','log'); set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('\Delta\mu');  xlim([min(delta_E),max(delta_E)]); ylim([min(delta_mu),max(delta_mu)]);
% title('P_2/P_1 (Matteo - Shiraishi)');
% min(min(ratio))
% 
% figure
% surf(E,MU,(EP-EP_Matteo)','EdgeColor','none'); 
% xlabel('\DeltaE');  ylabel('\Delta\mu');  xlim([min(delta_E),max(delta_E)]); ylim([min(delta_mu),max(delta_mu)]);
% title('EP(Total)- EP(Matteo)');
% 
% figure
% surf(E,MU,ratio_p_st','EdgeColor','none'); 
% %set(gca,'XScale','log'); set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('\Delta\mu');  xlim([min(delta_E),max(delta_E)]); ylim([min(delta_mu),max(delta_mu)]);
% title('P_2^{ST}/P_1^{ST}');
% 
% 
% figure
% diff(abs(diff)<1e-9)=0;
% surf(E,MU,real(diff)','EdgeColor','none'); 
% %set(gca,'XScale','log'); set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('\Delta\mu');  xlim([min(delta_E),max(delta_E)]); ylim([min(delta_mu),max(delta_mu)]);
% min(min(diff))
% title('EP(Matteo) - EP(Shiraishi)');
% 
% plot_results_as_function_of_deltaE_and_delta_Mu(delta_E,delta_mu,r,q,EP,f);
% 
% %% change delta_E and q. 
% delta_mu=log(400)
% r=100;
% 
% delta_E=(linspace(-8,8,400));
% q=logspace(-6,6,200);
% EP=0; 
% 
% for E_ind=1:length(delta_E)
%     for q_ind=1:length(q)
%         W=make_transition_matrix(r,q(q_ind),delta_E(E_ind),delta_mu,r_prime,q_prime,delta_mu_prime);
%         P_ss = find_ss(W);
%         
%         %W_st=find_stalling_matrix(r,q,delta_E(E_ind),delta_mu(mu_ind),r_prime);
%         W_st=make_transition_matrix(0,q(q_ind),delta_E(E_ind),delta_mu,r_prime,q_prime,delta_mu_prime); %same but with r=0
%         P_ss_st = find_ss(W_st);
%         
% %         close all
% %         figure; plot(P_ss); hold on; plot(P_ss_st,':')
% %         title(['P_{ss}(2)/P_{ss}(1)=' num2str(P_ss(2)/P_ss(1))   '  P_{ss}^{ST}(2)/P_{ss}^{ST}(1)=' num2str(P_ss_st(2)/P_ss_st(1))])
% 
%         ratio(E_ind,q_ind) = P_ss_st(2)/P_ss_st(1)  - P_ss(2)/P_ss(1);
%         ratio_p_st(E_ind,q_ind)=P_ss_st(2)/P_ss_st(1) ;
%         
%         EP(E_ind,q_ind)=calculate_entropty_production_ss(W,P_ss);  
%         EP_Shiraishi(E_ind,q_ind)=calculate_EP_Masked(W,P_ss,link);
%         EP_Matteo(E_ind,q_ind)=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
%         
%         %check that the sum of Shiraishi's over all links is the total EP
%         if abs(check_calculation(W,P_ss,EP(E_ind,q_ind)))>1e-6; 
%             %error('error'); 
%         end
%         diff(E_ind,q_ind)=EP_Matteo(E_ind,q_ind)-EP_Shiraishi(E_ind,q_ind);
%         if diff(E_ind,q_ind)<0
%             diff(E_ind,q_ind)
%         end
%     end
% end
% 
% [E,Q]=meshgrid(delta_E,q);
% figure
% surf(E,Q,real(ratio)','EdgeColor','none'); 
% %set(gca,'XScale','log'); 
% set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('q');  xlim([min(delta_E),max(delta_E)]); ylim([min(q),max(q)]);
% title('P_2/P_1 (Matteo - Shiraishi)');
% min(min(ratio))
% 
% figure
% surf(E,Q,ratio_p_st','EdgeColor','none'); 
% %set(gca,'XScale','log'); 
% set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('q');  xlim([min(delta_E),max(delta_E)]); ylim([min(q),max(q)]);
% title('P_2^{ST}/P_1^{ST}');
% 
% figure
% surf(E,Q,ratio_p_st','EdgeColor','none'); 
% %set(gca,'XScale','log'); 
% set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('q');  xlim([min(delta_E),max(delta_E)]); ylim([min(q),max(q)]);
% title('P_2^{ST}/P_1^{ST}');
% 
% figure
% diff(abs(diff)<1e-9)=0;
% surf(E,Q,(EP-EP_Matteo)','EdgeColor','none'); 
% %set(gca,'XScale','log'); 
% set(gca,'YScale','log'); 
% xlabel('\DeltaE');  ylabel('q');  xlim([min(delta_E),max(delta_E)]); ylim([min(q),max(q)]);
% min(min(diff))
% title('EP(total)-EP(Matteo)');

%% functions

function avg_step_time=calculate_avg_step_time(W)
    N=length(W); %size of the matrix
    omega=-diag(W); %escape rates
    P_ss = find_ss(W);
    for i=1:N
        R(i)=P_ss(i)*omega(i);
    end
    R=R/sum(R);
    avg_step_time=0;
    for i=1:N
        avg_step_time = avg_step_time + R(i)/omega(i);
    end
end

function EP_per_step=calculate_entropy_production_per_step(W)
    N=length(W); %size of the matrix
    omega=-diag(W); %escape rates
    P_ss = find_ss(W);
    
    for i=1:N
        for j=1:N
           P_matrix(i,j) = W(i,j)/omega(j);
           if i==j
               P_matrix(i,j) = 0;
           end
        end
        R(i)=P_ss(i)*omega(i);
    end
    R=R/sum(R);
    EP_per_step=0;
    for i=1:N
        for j=1:N
           if  P_matrix(j,i)~=0 && P_matrix(i,j)~=0
            EP_per_step = EP_per_step + R(i)*P_matrix(j,i)*log(P_matrix(j,i)/P_matrix(i,j));
           end
        end
    end
    
    
    
end

function [d,affinity]=Juan_Parrondo_Entropy(W,s,Psi_H,t);
    %syms s Psi_H t
    N=length(W); %size of the matrix
    omega=-diag(W); %escape rates

    
    p12=-W(1,2)/W(2,2); pH22=1-p12;
    p21=-W(2,1)/W(1,1); pH11=1-p21;
    
    
    pH1=squeeze(W(3:end,1)./omega(1)); pH1=pH1/sum(pH1);
    pH2=squeeze(W(3:end,2)./omega(2)); pH2=pH2/sum(pH2);
    
    for i=1:N-2
        for j=1:N-2
            Psi_H(i,j) = W(i+2,j+2)/(omega(j+2)+s);
            if i==j
                Psi_H(i,j) = 0;
            end
        end
    end

    a=inv(diag(ones(1,N-2))-Psi_H);
    
    for i=1:N-2
        v1H(i)=W(1,i+2)/(omega(i+2)+s);
        v2H(i)=W(2,i+2)/(omega(i+2)+s);
    end
    
    Psi_1H1=ilaplace(v1H*a*pH1);
    Psi_1H2=ilaplace(v1H*a*pH2);
    Psi_2H1=ilaplace(v2H*a*pH1);
    Psi_2H2=ilaplace(v2H*a*pH2);
    
    p1H1=double(int(Psi_1H1,t,0,Inf));
    p1H2=double(int(Psi_1H2,t,0,Inf));
    p2H1=double(int(Psi_2H1,t,0,Inf));
    p2H2=double(int(Psi_2H2,t,0,Inf));
    
    Psi_2H1_normalized = Psi_2H1/p2H1;
    Psi_1H2_normalized = Psi_1H2/p1H2;
    
    %figure; ezplot(Psi_1H2_normalized,[0,0.0001]); %for plotting the WTD
    
    R1=p12+p1H2*pH22;
    R2=p21+p2H1*pH11;
    RH1=pH11*R1;
    RH2=pH22*R2;
    sumR=R1+R2+RH1+RH2;
    
    R1=R1/sumR;     R2=R2/sumR;
    RH1 = RH1/sumR; RH2 = RH2/sumR;
    
    tau1 = -1/W(1,1);
    tau2 = -1/W(2,2);
    tauH1 = - double(subs(diff(v1H*a*pH1 + v2H*a*pH1, s),0));
    tauH2 = - double(subs(diff(v1H*a*pH2 + v2H*a*pH2 ,s),0));
    timePerStep = tau1*R1 + tau2*R2 + tauH1*RH1 + tauH2*RH2;
    
    js=R1*p21-R2*p12;
    
    d1=js * log ( p21*p1H2*pH22 /(p12*p2H1*pH11) );
    d2=R1*pH11*p2H1*double(int(Psi_2H1_normalized*log(Psi_2H1_normalized/Psi_1H2_normalized),t,0,Inf));
    d3=R2*pH22*p1H2*double(int(Psi_1H2_normalized*log(Psi_1H2_normalized/Psi_2H1_normalized),t,0,Inf));
    
    d=(d1+d2+d3)/timePerStep;
    affinity = d1/timePerStep;
end


function EP=calculate_EP_Masked(W,P_ss,link);
    EP=0;
    EP=   (W(link(1),link(2))*P_ss(link(2))  - W(link(2),link(1))*P_ss(link(1))  )*log(W(link(1),link(2)) * P_ss(link(2)) / W(link(2),link(1)) / P_ss(link(1)) );
end

function EP=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
    EP=0;
    EP=   (W(link(1),link(2))*P_ss(link(2))  - W(link(2),link(1))*P_ss(link(1))  )*log(W(link(1),link(2)) * P_ss_st(link(2)) / W(link(2),link(1)) / P_ss_st(link(1)) );
end

function EP=calculate_EP_total_minus_Matteo(W,P_ss,P_ss_st)
% link=[1,2];
% EP_1=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
link=[2,3];
EP_2=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
link=[3,4];
EP_3=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
link=[4,1];
EP_4=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
link=[1,3];
EP_5=calculate_EP_Marginal(W,P_ss,P_ss_st,link);
EP=EP_2+EP_3+EP_4+EP_5;
end


function Z = find_ss(W);
temp=null(W); %find eigenvector of steady state
% if size(temp,2)>1 
%     if size(temp,2)>2
%         error('error'); 
%     else %size(temp,2)=2
%         % If the null space is of rank 2, just take a linear combination of
%         % the 2 eigenvectors
%         temp(:,1)=temp(:,1)/sum(temp(:,1));
%         temp(:,2)=temp(:,2)/sum(temp(:,2));
%         temp=0.2*temp(:,1)+0.8*temp(:,2);
%     end
% end
if size(temp,2)~=1 
    error('error'); 
end;
Z = temp/sum(temp);
end

function W=make_transition_matrix(r,q,delta_E,delta_mu,R,q_prime,delta_mu_prime)
% exit rates
w_12=r;
w_21=r*exp(-delta_E);
w_14=q;
w_41=q*exp(delta_mu);
w_32=q;
w_23=q*exp(delta_mu);
w_34=R;
w_43=R*exp(-delta_E);

w_13=q_prime;
w_31=q_prime*exp(delta_mu_prime);

l_1=w_21+w_31+w_41;
l_2=w_12+w_32;
l_3=w_13+w_23+w_43;
l_4=w_14+w_34;

W=[ -l_1   w_12       w_13     w_14;...
    w_21   -l_2       w_23     0;...
    w_31   w_32       -l_3     w_34;...
    w_41   0          w_43    -l_4];

% W=[ -l_1             r                  0                   q;...
%     r*exp(-delta_E)  -r-q*exp(delta_mu) q                   0;...
%     0                q*exp(delta_mu)    -R*exp(-delta_E)-q  R;...
%     q*exp(delta_mu)  0                  R*exp(-delta_E)     -l_4];
% 
% W=[ -r*exp(-delta_E)-q  r                  0                   q*exp(delta_mu);...
%     r*exp(-delta_E)     -r-q*exp(delta_mu) q                   0;...
%     0                   q*exp(delta_mu)    -R*exp(-delta_E)-q  R;...
%     q                   0                  R*exp(-delta_E)     -q*exp(delta_mu)-R];

% W=[ -q*exp(delta_mu)-r  r*exp(-delta_E)     0                   q;...
%     r                   -r*exp(-delta_E)-q  q*exp(delta_mu)     0;...
%     0                   q                   -l_3                R;...
%     q*exp(delta_mu)     0                   R*exp(-delta_E)    -l_4];

% W=[ -l_1             r          0                   q;...
%     r*exp(-delta_E)  -l_2       q*exp(delta_mu)     0;...
%     0                q          -q*exp(delta_mu)-R  R*exp(-delta_E);...
%     q*exp(delta_mu)  0          R                   -R*exp(-delta_E)-q];

% W=[ -q*exp(delta_mu)-r  r*exp(-delta_E)      0                   q;...
%     r                   -r*exp(-delta_E)-q   q*exp(delta_mu)     0;...
%     0                   q                   -q*exp(delta_mu)-R  R*exp(-delta_E);...
%     q*exp(delta_mu)     0                   R                   -R*exp(-delta_E)-q];

end

function W=find_stalling_matrix(r,q,delta_E,delta_mu,R)

% exit rates
l_1=q*exp(delta_mu);
l_2=q;
l_3=R*exp(-delta_E)+q*exp(delta_mu);
l_4=R+q;

W=[ -l_1             0      0                   q;...
    0                -l_2   q*exp(delta_mu)     0;...
    0                q      -l_3                R;...
    q*exp(delta_mu)  0      R*exp(-delta_E)    -l_4];

% W=[ -l_1             0                  0                   q;...
%     0               -q*exp(delta_mu)    q                   0;...
%     0                q*exp(delta_mu)    -R*exp(-delta_E)-q  R;...
%     q*exp(delta_mu)  0                  R*exp(-delta_E)     -l_4];
% 
% W=[ -q     0                    0                   q*exp(delta_mu);...
%     0     -q*exp(delta_mu)      q                   0;...
%     0      q*exp(delta_mu)      -R*exp(-delta_E)-q  R;...
%     q      0                    R*exp(-delta_E)     -q*exp(delta_mu)-R];

% W=[ -l_1             0      0                   q;...
%     0                -l_2   q*exp(delta_mu)     0;...
%     0                q      -R-q*exp(delta_mu)  R*exp(-delta_E);...
%     q*exp(delta_mu)  0      R                   -R*exp(-delta_E)-q];
end


% function W=make_transition_matrix(r,q,delta_E,delta_mu)
% % exit rates
% l_1=r*exp(-delta_E)+q*exp(delta_mu);
% l_2=r+q;
% W=[ -l_1             r      0                   q;...
%     r*exp(-delta_E)  -l_2   q*exp(delta_mu)     0;...
%     0                q      -l_1                r;...
%     q*exp(delta_mu)  0       r*exp(-delta_E)    -l_2];
% end

% function W=find_stalling_matrix(r,q,delta_E,delta_mu)
% % exit rates
% l_1=q*exp(delta_mu);
% l_2=q;
% W=[ -l_1             0      0                   q;...
%     0               -l_2    q*exp(delta_mu)     0;...
%     0                q      -l_1                0;...
%     q*exp(delta_mu)  0      0                   -l_2];
% end


%This is a simpler expression for steady state
function EP=calculate_entropty_production_ss(W,p_ss)
EP=0;
for i=1:4
    for j=1:4
        if and(W(i,j)~=0,W(j,i)~=0)
            EP=EP+W(i,j)*p_ss(j)*log( W(i,j)/W(j,i) );
        end
    end
end
end

% %This is a general expression (not nesseseraly for a steady state)
% function EP=calculate_entropty_production2(W,p_ss)
% EP=0;
% for i=1:4
%     for j=1:4
%         if and(W(i,j)~=0,W(j,i)~=0)
%             EP=EP+1/2*(W(i,j)*p_ss(j) - W(j,i)*p_ss(i))*log( W(i,j)*p_ss(j)/W(j,i)/p_ss(i) );
%         end
%     end
% end
% end


function res=check_calculation(W,P_ss,EP)
link=[1,2];
EP_Shiraishi1=calculate_EP_Masked(W,P_ss,link);
link=[2,3];
EP_Shiraishi2=calculate_EP_Masked(W,P_ss,link);
link=[3,4];
EP_Shiraishi3=calculate_EP_Masked(W,P_ss,link);
link=[4,1];
EP_Shiraishi4=calculate_EP_Masked(W,P_ss,link);
link=[1,3];
EP_Shiraishi5=calculate_EP_Masked(W,P_ss,link);
res=(EP-(EP_Shiraishi1+EP_Shiraishi2+EP_Shiraishi3+EP_Shiraishi4+EP_Shiraishi5))/EP;

%if EP<1e-3 res=0; end
if res<1e-3 res=0; end
end

function EP_Shiraishi_link=calculate_Shiraishi_for_all_links(W,P_ss)
link=[1,2];
EP_Shiraishi_link(1)=calculate_EP_Masked(W,P_ss,link);
link=[2,3];
EP_Shiraishi_link(2)=calculate_EP_Masked(W,P_ss,link);
link=[3,4];
EP_Shiraishi_link(3)=calculate_EP_Masked(W,P_ss,link);
link=[4,1];
EP_Shiraishi_link(4)=calculate_EP_Masked(W,P_ss,link);
link=[1,3];
EP_Shiraishi_link(5)=calculate_EP_Masked(W,P_ss,link);
end


function f=plot_results_as_function_of_rates(delta_E,delta_mu,r,q,EP)
[R,Q]=meshgrid(r,q);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f=figure('position',[10,10, 900,800]);
subplot(2,2,1)
surf(R,Q,EP','EdgeColor','none');
set(gca,'XScale','log'); set(gca,'YScale','log');
xlabel('r');  ylabel('q'); 
xlim([min(r),max(r)]); ylim([min(q),max(q)]); 
title('Entropy Production ')
set(gca,'XTick',logspace(-3,3,7)); set(gca,'YTick',logspace(-3,3,7));
title(['Entropy Production (e^{-\DeltaE}=' num2str(exp(-delta_E)) '  e^{\Delta\mu}=' num2str(exp(delta_mu)) ')'])


subplot(2,2,2)
h=pcolor(r,q,log(EP')); 
set(h, 'EdgeColor', 'none');
set(gca,'XScale','log'); set(gca,'YScale','log');
xlabel('r');  ylabel('q'); 
set(gca,'XTick',logspace(-3,4,8)); set(gca,'YTick',logspace(-3,4,8));
colorbar;
title('Entropy Production (log scale)')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function plot_results_as_function_of_deltaE_and_delta_Mu(delta_E,delta_mu,r,q,EP,f)
figure(f);
[E,MU]=meshgrid(delta_E,delta_mu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure('position',[100,500, 900,300])
subplot(2,2,3)
surf(E,MU,abs(EP)','EdgeColor','none');
%set(gca,'XScale','log'); set(gca,'YScale','log');
xlabel('\DeltaE');  ylabel('\Delta\mu'); 
xlim([min(delta_E),max(delta_E)]); ylim([min(delta_mu),max(delta_mu)]); 
title(['Entropy Production (r=' num2str(r) ' q=' num2str(q) ')'])
%set(gca,'XTick',logspace(-1,3,5)); set(gca,'YTick',logspace(-1,3,5));

subplot(2,2,4)
h=pcolor(delta_E,delta_mu,(EP')); 
set(h, 'EdgeColor', 'none');
%set(gca,'XScale','log'); set(gca,'YScale','log');
xlabel('\DeltaE');  ylabel('\Delta\mu'); 
%set(gca,'XTick',logspace(-1,3,5)); set(gca,'YTick',logspace(-1,3,5));
colorbar;
title('Entropy Production')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end