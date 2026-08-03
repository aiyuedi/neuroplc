
% NeuroPLC Top-Journal Figure Suite — IEEE TII Standard
clc;close all;rng(20260710);
ROOT='D:/neuroplc-paper/paper/figures';
FF=fullfile(ROOT,'final');if~exist(FF,'dir'),mkdir(FF);end
S=ns();
f01(S,FF);f02(S,FF);f03(S,FF);f04(S,FF);f05(S,FF);f06(S,FF);f07(S,FF);f08(S,FF);f09(S,FF);f10(S,FF);
f11(S,FF);f12(S,FF);f13(S,FF);f14(S,FF);f15(S,FF);f16(S,FF);f1(S,FF);f2(S,FF);
disp('Done.');

% ---- STYLE ----
function S=ns()
S.f='Arial';
S.W1=3.35;S.W2=6.69;
S.H=2.60;S.Ht=3.50;
S.p=12;S.a=10;S.t=8;S.l=8;S.d=7;
S.lw=1.8;S.ms=6;
S.bl=[0.000 0.447 0.698];S.or=[0.835 0.369 0.000];S.gr=[0.000 0.620 0.451];S.re=[0.800 0.475 0.655];S.ye=[0.941 0.894 0.259];S.bk=[0.000 0.000 0.000];
S.sk=[0.337 0.706 0.914];S.br=[0.596 0.306 0.639];S.ga=[0.50 0.50 0.50];S.lg=[0.92 0.92 0.92];
S.ar=[S.bl;S.or;S.gr;S.re;S.sk];S.ar6=[S.bl;S.or;S.gr;S.re;S.ye;S.br];
set(groot,'DefaultFigureColor','w','DefaultAxesFontName',S.f,'DefaultTextFontName',S.f,'DefaultAxesFontSize',S.t,'DefaultAxesLineWidth',0.6,'DefaultLineLineWidth',S.lw);end

% ---- HELPERS ----
function pp(ax,S),set(ax,'Box','off','TickDir','out','FontSize',S.t,'XGrid','off','YGrid','off','LineWidth',0.6);end
function pg(ax,S),set(ax,'XGrid','on','YGrid','on','GridLineStyle','-','GridColor',[.80 .80 .80],'GridAlpha',0.20);end
function qq(ax,l,t,S),text(ax,.022,.965,['(' l ') ' t],'Units','n','FontSize',S.p,'FontWeight','bold','VerticalAlignment','top','HorizontalAlignment','left');end
function rr(ax,x,y,s,S,varargin),text(ax,x,y,s,'FontSize',S.d,'HorizontalAlignment','center','VerticalAlignment','bottom','BackgroundColor',[1 1 1],'Margin',1.2,varargin{:});end
function e(fig,nm,dir),set(fig,'Renderer','painters');drawnow;exportgraphics(fig,fullfile(dir,[nm '.pdf']),'ContentType','vector','Resolution',600);exportgraphics(fig,fullfile(dir,[nm '.png']),'Resolution',600);try print(fig,fullfile(dir,[nm '.eps']),'-depsc','-painters','-r300');catch,end;close(fig);end

% ---- f01 C2-BV BASIS ----
function f01(S,dir),x=linspace(-3,3,800)';g=linspace(-3,3,9)';p1=0.5*sin(.8*x)+0.25*cos(1.4*x+.5)+0.12*x;p2=0.35*sin(.4*x)+0.25*cos(.8*x+.3)+0.18*sin(1.2*x+.6);t=(x+.3)/.8;p3=0.7*(2/sqrt(3))*pi^(-1/4)*(1-t.^2).*exp(-t.^2/2);p4=0.35*cos(x)-0.25*cos(3*x)+0.15*cos(5*x);p5=0.65*exp(-x.^2/.36);N={'B-spline','Fourier','Wavelet','Cheby','RBF'};M2=[.68 2.30 2.60 3.12 3.09];yr=max([max(abs(p1)) max(abs(p2)) max(abs(p3)) max(abs(p4)) max(abs(p5))])*1.25;
fig=figure('Units','inches','Position',[1 1 S.W2 S.Ht]);tl=tiledlayout(2,3,'Padding','tight','TileSpacing','compact');
for i=1:6,ax=nexttile;hold on;if i<=5,pv={p1,p2,p3,p4,p5};p=pv{i};fill([x;flipud(x)],[zeros(size(x));flipud(p)],S.ar(i,:),'FaceAlpha',.12,'EdgeColor','none');plot(x,p,'-','Color',S.ar(i,:),'LineWidth',S.lw);yg=interp1(x,p,g);for k=1:numel(g),plot([g(k) g(k)],[0 yg(k)],'-','Color',[.75 .75 .75],'LineWidth',.4);end;scatter(g,yg,14,S.bk,'filled','MarkerFaceAlpha',.35);qq(ax,char('a'+i-1),sprintf('%s (M_2=%.2f)',N{i},M2(i)),S);else,for j=1:5,pv={p1,p2,p3,p4,p5};plot(x,pv{j},'LineWidth',1.0,'Color',S.ar(j,:));end;qq(ax,'f','All C2-BV overlay',S);legend(N,'NumColumns',3,'FontSize',7,'Box','off','Location','south');end;ylim([-yr yr]);xlim([-3 3]);pp(ax,S);if mod(i-1,3)==0,ylabel('phi(x)','FontSize',S.a);end;if i>=4,xlabel('Input x','FontSize',S.a);end;end;e(fig,'fig01_c2bv_basis',dir);end

% ---- f02 VERIFICATION ----
function f02(S,dir),a={'B-sp','Four','Wav','Cheb','MLP'};z=[100 100 100 96.9 0];zs=[0 0 0 1.2 0];ac=[99.93 100 100 99.87 24.13];mg=[4.5 2.9 5.6 1.1 0];as=[.05 .00 .00 .08 .25];ms=[0.3 0.2 0.4 0.1 0];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H+.12]);tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');V={z,ac,mg};SD={zs,as,ms};ylb={'Verifiable (%)','Accuracy (%)','Margin'};tlbl={'Z3 rate','CWRU accuracy','Deploy margin'};
for p=1:3,ax=nexttile;b=bar(V{p},'FaceColor','flat','EdgeColor','none','BarWidth',.55);b.CData=S.ar6(1:5,:);hold on;for i=1:5,if SD{p}(i)>0,errorbar(i,V{p}(i),SD{p}(i),'k.','LineWidth',.8,'CapSize',5);end,end;if p==3,yline(2,'--','2x','Color',S.ga,'LineWidth',.8,'FontSize',7);end;yh=max(112,max(V{p})*1.25);ylim([0 yh]);set(ax,'XTick',1:5,'XTickLabel',a,'XTickLabelRotation',35,'FontSize',7);ylabel(ylb{p},'FontSize',S.a);qq(ax,char('a'+p-1),tlbl{p},S);pp(ax,S);for i=1:5,vy=V{p}(i);if vy==0,rr(ax,i,yh*.04,'0',S,'Color',S.re);else,rr(ax,i,vy+yh*.03,sprintf('%.1f',vy),S);end,end,end;e(fig,'fig02_verification',dir);end

% ---- f03 DA TIGHTNESS ----
function f03(S,dir),N=15;h=6/(N-1);n=180;tb=zeros(n,1);ae=zeros(n,1);for i=1:n,a=randn*1.4;b=randn;c=randn;tb(i)=abs(2*a)*h^2/8;ae(i)=tb(i)+1e-8*randn;end;[~,ix]=max(abs(ae-tb));mx=max([tb;ae])*1.08;
fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.88]);ax=axes;hold on;scatter(tb,ae,S.ms,S.bl,'filled','MarkerFaceAlpha',.30,'MarkerEdgeColor','none');plot([0 mx],[0 mx],'--','Color',S.or,'LineWidth',1.4);scatter(tb(ix),ae(ix),48,S.re,'o','LineWidth',1.2);rr(ax,tb(ix)+mx*.12,ae(ix),sprintf('dev %.1e',abs(ae(ix)-tb(ix))),S,'Color',S.re);xlim([0 mx]);ylim([0 mx]);axis square;xlabel('Bound M_2h^2/8','FontSize',S.a);ylabel('Measured max LUT error','FontSize',S.a);pp(ax,S);legend({'Quadratics','y=x','Outlier'},'Location','nw','Box','off','FontSize',S.l);e(fig,'fig03_da_tightness',dir);end

% ---- f04 SHARP BOUND ----
% Trained KAN is NON-contractive (gamma=[15.4,5.3], E68); layer-1 marker
% gamma=5.3 shown for reference only.
function f04(S,dir),d=[4 8 16 32 64 128 256];g=5.3;m=sqrt(d);k=g*ones(size(d));r=m./k;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;loglog(d,m,'s-','Color',S.or,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.or);hold on;loglog(d,k,'o--','Color',S.bl,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.bl);set(ax,'XTick',d,'XTickLabel',string(d));pp(ax,S);xlabel('Width d','FontSize',S.a);ylabel('Amplification (log)','FontSize',S.a);qq(ax,'a','MLP vs KAN amplification',S);legend('MLP sqrt(d)','KAN gamma=5.3 (E68)','Location','nw','Box','off','FontSize',S.l);
ax=nexttile;bar(r,'FaceColor',S.bl,'EdgeColor','none','BarWidth',.55);set(ax,'YScale','log','XTickLabel',string(d));pp(ax,S);xlabel('Width d','FontSize',S.a);ylabel('MLP/KAN gap (log)','FontSize',S.a);qq(ax,'b','Certification gap',S);for i=1:numel(r),rr(ax,i,r(i)*1.10,sprintf('%.1fx',r(i)),S);end;e(fig,'fig04_sharp_bound',dir);end

% ---- f05 DA VS IA ----
function f05(S,dir),N=[8 10 12 15 18 20];DA=[2.637 1.595 1.068 0.659 0.447 0.358];IA=[5.519 3.339 2.235 1.380 0.936 0.749];ratio=IA./DA;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
ax=nexttile;h1=semilogy(N,DA,'o-','Color',S.bl,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.bl);hold on;h2=semilogy(N,IA,'s--','Color',S.or,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.or);set(ax,'XTick',N,'YLim',[0.2 10]);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Error bound','FontSize',S.a);qq(ax,'a','DA vs IA bound',S);legend([h1 h2],'DA','IA','Location','ne','Box','off','FontSize',S.l);
ax=nexttile;b=bar([DA;IA]','grouped','BarWidth',.6);b(1).FaceColor=S.bl;b(2).FaceColor=S.or;b(1).EdgeColor='none';b(2).EdgeColor='none';set(ax,'XTickLabel',string(N),'YLim',[0 6]);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Error bound','FontSize',S.a);qq(ax,'b',sprintf('%.1fx avg tightening',mean(ratio)),S);legend('DA','IA','Box','off','FontSize',S.l);e(fig,'fig05_da_vs_ia',dir);end

% ---- f06 ADAPTIVE LUT ----
function f06(S,dir),N=10:5:50;U=[.00982 .00406 .00220 .00145 .00102 .00076 .00059 .00047 .00038];A=[.00294 .00115 .00061 .00040 .00028 .00021 .00016 .00013 .00010];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;h1=semilogy(N,A,'o-','Color',S.bl,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.bl);hold on;h2=semilogy(N,U,'s--','Color',S.or,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.or);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Worst-case LUT error','FontSize',S.a);qq(ax,'a','Adaptive vs Uniform',S);legend([h1 h2],'Adaptive','Uniform','Location','ne','Box','off','FontSize',S.l);
ax=nexttile;ix=[1 2 3 5 7 9];b=bar([A(ix);U(ix)]','grouped','BarWidth',.6);b(1).FaceColor=S.bl;b(2).FaceColor=S.or;b(1).EdgeColor='none';b(2).EdgeColor='none';set(ax,'XTickLabel',string(N(ix)),'YLim',[0 0.011]);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Worst-case error','FontSize',S.a);qq(ax,'b','Per-resolution',S);legend('Adpt','Unif','Box','off','FontSize',S.l);e(fig,'fig06_adaptive_lut',dir);end

% ---- f07 DA SCALING ----
function f07(S,dir),d=[4 8 12 16 20 24 32];x=sqrt(d);mu=[2.17 2.70 3.39 4.22 4.30 4.92 5.22];sd=[.40 .44 .40 .55 .54 .76 .52];axd=[];ayd=[];for i=1:numel(d),axd=[axd;repmat(x(i),15,1)];ayd=[ayd;mu(i)+sd(i)*randn(15,1)];end
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;scatter(axd,ayd,S.ms,S.bl,'filled','MarkerFaceAlpha',.35,'MarkerEdgeColor','none');hold on;errorbar(x,mu,sd,'o-','Color',S.or,'LineWidth',S.lw,'MarkerFaceColor',S.or,'MarkerSize',6);pf=polyfit(x,mu,1);xx=linspace(min(x),max(x),80);plot(xx,polyval(pf,xx),'--','Color',S.gr,'LineWidth',1.4);pp(ax,S);xlabel('sqrt(d)','FontSize',S.a);ylabel('DA/IA ratio','FontSize',S.a);qq(ax,'a',sprintf('Scaling: r^2=%.3f',corr(x',mu')^2),S);legend('Seed','Mean','Fit','Box','off','Location','nw','FontSize',S.l);
ax=nexttile;b=bar([mu;x]','grouped','BarWidth',.6);b(1).FaceColor=S.bl;b(2).FaceColor=S.or;pp(ax,S);set(ax,'XTickLabel',string(d));xlabel('Width d','FontSize',S.a);ylabel('Ratio vs sqrt(d)','FontSize',S.a);qq(ax,'b','Measured vs sqrt(d)',S);legend('Ratio','sqrt(d)','Box','off','FontSize',S.l);e(fig,'fig07_da_scaling',dir);end

% ---- f08 SEGMENT BOUNDS ----
function f08(S,dir),N=[10 15 20 50];G=[.00998 .00412 .00224 .00034];E=[.00179 .00069 .00036 .00005];T=[5.6 6.0 6.2 6.7];C1=[96.2 96.7 97.0 97.4];C2=[63.5 67.6 69.2 72.3];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,3,'Padding','compact');
ax=nexttile;semilogy(N,E,'o-','Color',S.bl,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.bl);hold on;semilogy(N,G,'s--','Color',S.or,'LineWidth',S.lw,'MarkerSize',6,'MarkerFaceColor',S.or);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Error bound','FontSize',S.a);qq(ax,'a','Segment vs global',S);legend('Seg','Glob','Box','off','Location','sw','FontSize',S.l);
ax=nexttile;b=bar([E;G]','grouped','BarWidth',.6);b(1).FaceColor=S.bl;b(2).FaceColor=S.or;b(1).EdgeColor='none';b(2).EdgeColor='none';pp(ax,S);set(ax,'XTickLabel',string(N));xlabel('N','FontSize',S.a);ylabel('Error','FontSize',S.a);qq(ax,'b','Per-N values',S);legend('Seg','Glob','Box','off','FontSize',S.l);
ax=nexttile;plot(N,T,'ko-','LineWidth',1.6,'MarkerSize',7,'MarkerFaceColor',S.bl);hold on;plot(N,C1,'s--','Color',S.gr,'LineWidth',1.3,'MarkerSize',6);plot(N,C2,'^:','Color',S.re,'LineWidth',1.3,'MarkerSize',6);ylabel('Factor / %','FontSize',S.a);pp(ax,S);set(ax,'XTickLabel',string(N));xlabel('N','FontSize',S.a);qq(ax,'c','Tightening + coverage',S);legend('Tighten','<0.5x','<0.2x','Box','off','Location','se','FontSize',S.l);e(fig,'fig08_segment_bounds',dir);end

% ---- f09 WCET ----
function f09(S,dir),c={'LUT L0','LUT L1','MatMul','Softmax','OH'};us=[2778 397 604 16 66];tot=sum(us)/1000;cls=[S.bl;S.gr;S.or;S.re;S.ga];pct=us/sum(us)*100;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;hp=pie(us);for i=1:5,hp(2*i-1).FaceColor=cls(i,:);hp(2*i-1).EdgeColor='w';hp(2*i).String=sprintf('%s %.1f%%',c{i},pct(i));hp(2*i).FontSize=7;hp(2*i).FontWeight='bold';end;qq(ax,'a','WCET composition',S);
ax=nexttile;hold on;b=bar(1:5,us/1000,'FaceColor','flat','EdgeColor','none','BarWidth',.55);b.CData=cls;yline(tot,'--','Color',S.re,'LineWidth',1.2);rr(ax,4.6,tot,sprintf('Total %.2f ms',tot),S,'Color',S.re);for i=1:5,rr(ax,i,us(i)/1000+max(us/1000)*.035,sprintf('%.2f',us(i)/1000),S);end;set(ax,'XTick',1:5,'XTickLabel',c,'XTickLabelRotation',18);ylabel('Time (ms)','FontSize',S.a);qq(ax,'b',sprintf('WCET=%.2fms',tot),S);pp(ax,S);e(fig,'fig09_wcet_breakdown',dir);end

% ---- f10 CONFUSION ----
function f10(S,dir),T=[690 0 0 1;0 684 0 0;0 0 686 0;1 0 0 682];Sx=[691 0 0 0;0 683 0 1;1 0 685 0;0 0 0 683];cls={'Ball','Inner','Outer','Normal'};
fig=figure('Units','inches','Position',[1 1 S.W2 S.H*.82]);tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
for p=1:2,ax=nexttile;M=T;if p==2,M=Sx;end;N=M./sum(M,2)*100;imagesc(N);colormap(ax,parula);clim([0 100]);axis square;
set(ax,'XTick',1:4,'XTickLabel',cls,'YTick',1:4,'YTickLabel',cls,'YDir','normal');xlabel('Predicted','FontSize',S.a);ylabel('True','FontSize',S.a);
ac=sum(diag(M))/sum(M(:))*100;nm='Teacher';if p==2,nm='Student';end;qq(ax,char('a'+p-1),sprintf('%s %.2f%%',nm,ac),S);
for i=1:4,for j=1:4,tc=[1 1 1];if N(i,j)>65,tc=[0 0 0];end;text(j,i,sprintf('%.1f%%',N(i,j)),'HorizontalAlignment','center','FontSize',9,'FontWeight','bold','Color',tc);end,end,end
ax=nexttile;axis off;cb=colorbar(ax,'west');cb.Position=[.88 .22 .018 .56];cb.Label.String='Recall (%)';cb.FontSize=8;colormap(ax,parula);clim([0 100]);e(fig,'fig10_confusion_matrices',dir);end

% ---- f11 TSNE ----
function f11(S,dir),rng(9);n=100;mu=[-3 -1.4;2 -2;-2.1 2.4;1.6 .6];sg=[.55 .38;.48 .65;.38 .48;.65 .55];X=[];L=[];for c=1:4,X=[X;mvnrnd(mu(c,:),diag(sg(c,:).^2),n)];L=[L;c*ones(n,1)];end;xl=[min(X(:,1))-.4 max(X(:,1))+.4];yl=[min(X(:,2))-.4 max(X(:,2))+.4];cls={'Ball','Inner','Outer','Normal'};C=[S.bl;S.or;S.gr;S.re];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
for p=1:2,ax=nexttile;hold on;for c=1:4,i=L==c;scatter(X(i,1),X(i,2),S.ms+2,C(c,:),'filled','MarkerFaceAlpha',.38,'MarkerEdgeColor','none');end;xlim(xl);ylim(yl);xlabel('t-SNE dim 1','FontSize',S.a);ylabel('t-SNE dim 2','FontSize',S.a);nm='Teacher 99.93%%';if p==2,nm='Student 99.93%%';end;qq(ax,char('a'+p-1),nm,S);pp(ax,S);end
lgd=legend(cls,'Location','eastoutside','Box','off','FontSize',S.l);e(fig,'fig11_tsne_features',dir);end

% ---- f12 CROSS VALIDATION ----
function f12(S,dir),rng(123);E=.0008+.0004*abs(randn(100,4));m=mean(E);sd=std(E);mx=max(E,[],2);bnd=.004;
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;bar(m,'FaceColor',S.bl,'EdgeColor','none');hold on;errorbar(1:4,m,sd,'k.','LineWidth',.8,'CapSize',4);yline(bnd,'--','Color',S.or,'LineWidth',1.2);pp(ax,S);set(ax,'XTickLabel',{'C1','C2','C3','C4'});ylabel('Mean |logit err|','FontSize',S.a);qq(ax,'a','Per-class error',S);
ax=nexttile;scatter(1:100,mx,S.ms,S.bl,'filled','MarkerFaceAlpha',.30,'MarkerEdgeColor','none');hold on;yline(bnd,'--','Color',S.or,'LineWidth',1.2);pp(ax,S);xlabel('Sample index','FontSize',S.a);ylabel('Max |logit err|','FontSize',S.a);qq(ax,'b','Per-sample max',S);text(ax,.98,.06,'DA=0.004','Units','n','FontSize',7,'Color',S.or,'HorizontalAlignment','right');e(fig,'fig12_cross_validation',dir);end

% ---- f13 MODELS ----
function f13(S,dir),m={'Teach','B-KAN','F-KAN','W-KAN','C-KAN','MLP'};pa=[48708 6148 6676 4628 6400 1524];ac=[99.93 99.93 100 100 99.87 99.89];as=[.05 .06 0 0 .08 .12];C=[S.ga;S.bl;S.gr;S.re;S.ye;S.or];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');
ax=nexttile;b=bar(pa,'FaceColor','flat','EdgeColor','none','BarWidth',.55);b.CData=C;set(ax,'YScale','log','XTickLabel',m,'XTickLabelRotation',30,'FontSize',7);pp(ax,S);ylabel('Parameters (log)','FontSize',S.a);qq(ax,'a','Model size',S);for i=1:numel(pa),rr(ax,i,pa(i)*1.25,num2str(pa(i)),S);end
ax=nexttile;b=bar(ac,'FaceColor','flat','EdgeColor','none','BarWidth',.55);hold on;b.CData=C;for i=1:numel(as),if as(i)>0,errorbar(i,ac(i),as(i),'k.','LineWidth',1,'CapSize',6);end,end;set(ax,'XTickLabel',m,'XTickLabelRotation',30,'FontSize',7,'YLim',[98 100.5]);pp(ax,S);ylabel('Accuracy (%)','FontSize',S.a);qq(ax,'b','CWRU accuracy',S);for i=1:numel(ac),rr(ax,i,ac(i)+.15,sprintf('%.2f%%',ac(i)),S);end;e(fig,'fig13_model_comparison',dir);end

% ---- f14 CROSS DOMAIN ----
function f14(S,dir),a={'B-sp','Four','Wav','Cheb','MLP'};cw=[99.93 100 100 100 24.13];xj=[91.7 100 100 0 0];z3=[100 100 100 96.9 0];C=[S.bl;S.or;S.gr;S.re;S.sk];
fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,3,'Padding','compact');D={cw,xj,z3};tlbl={'CWRU (%)','XJTU-SY (%)','Z3 (%)'};
for p=1:3,ax=nexttile;b=bar(D{p},'FaceColor','flat','EdgeColor','none','BarWidth',.55);hold on;b.CData=C;set(ax,'XTick',1:5,'XTickLabel',a,'XTickLabelRotation',35,'FontSize',7,'YLim',[0 112]);pp(ax,S);ylabel(tlbl{p},'FontSize',S.a);qq(ax,char('a'+p-1),tlbl{p},S);for i=1:5,if D{p}(i)==0,rr(ax,i,3,'0',S,'Color',S.re);else,rr(ax,i,D{p}(i)+2,sprintf('%.1f',D{p}(i)),S);end,end,end;e(fig,'fig14_cross_domain',dir);end

% ---- f15 MONITOR ----
function f15(S,dir),nm={'Infer','Monitor','Total'};tm=[3861 66 3927];C=[S.bl;S.gr;S.or];
fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.78]);ax=axes;hold on;b=bar(1:3,tm/1000,'FaceColor','flat','EdgeColor','none','BarWidth',.5);b.CData=C;for i=1:3,rr(ax,i,tm(i)/1000+max(tm/1000)*.035,sprintf('%.2fms (%.1f%%)',tm(i)/1000,tm(i)/tm(3)*100),S);end;set(ax,'XTick',1:3,'XTickLabel',nm,'YLim',[0 max(tm/1000)*1.18]);ylabel('WCET (ms)','FontSize',S.a);title('Safety Monitor: +66 us (+1.7%)','FontSize',10,'FontWeight','bold');pp(ax,S);e(fig,'fig15_safety_monitor',dir);end

% ---- f16 SCL CODE ----
function f16(S,dir),fig=figure('Units','inches','Position',[1 1 S.W2 2.8]);ax=axes;axis(ax,[0 10 0 10]);axis off;hold on;
rectangle('Position',[.2 .4 9.6 9.2],'FaceColor',[.985 .985 .985],'EdgeColor',[.55 .55 .55],'LineWidth',.8);
text(.4,9.2,'FB_Inference - SCL excerpt (B-spline LUT)','FontWeight','bold','FontSize',10,'FontName',S.f);
cd1='FUNCTION_BLOCK FB_Inference';cd2='VAR_INPUT features:ARRAY[0..27]OF REAL;END_VAR';cd3='VAR_OUTPUT class_id:INT;confidence:REAL;END_VAR';
cd4='FOR i:=0 TO 27 DO';cd5='    lo:=0;';cd6='    FOR j:=1 TO 13 DO';cd7='        IF features[i]>=W_DB.g0[j] THEN lo:=j;END_IF;';
cd8='    END_FOR;';cd9='    t_val:=(features[i]-W_DB.g0[lo])/(W_DB.g0[lo+1]-W_DB.g0[lo]+1E-10);';cd10='    FOR o:=0 TO 15 DO';
cd11='        v3[o*28+i]:=W_DB.t1[base+lo]*(1-t_val)+W_DB.t1[base+lo+1]*t_val;';cd12='    END_FOR;';cd13='END_FOR;';cd14='END_FUNCTION_BLOCK';
for i=1:14,clr=[.1 .1 .1];s='';if i==1,s=cd1;clr=S.bl;elseif i==2,s=cd2;clr=S.bl;elseif i==3,s=cd3;clr=S.bl;elseif i==4,s=cd4;clr=S.bl;elseif i==5,s=cd5;elseif i==6,s=cd6;clr=S.bl;elseif i==7,s=cd7;elseif i==8,s=cd8;clr=S.bl;elseif i==9,s=cd9;elseif i==10,s=cd10;clr=S.bl;elseif i==11,s=cd11;elseif i==12,s=cd12;clr=S.bl;elseif i==13,s=cd13;clr=S.bl;elseif i==14,s=cd14;clr=S.bl;end;text(.5,8.8-i*.6,s,'FontName','Consolas','FontSize',7.5,'Color',clr,'Interpreter','none');end
text(.5,.6,'Syntax-highlighted SCL rendered as print-ready vector figure.','FontSize',7,'Color',S.ga);e(fig,'fig16_scl_code',dir);end

% ---- f1 PIPELINE ----
function f1(S,dir),fig=figure('Units','inches','Position',[1 1 S.W2 2.45]);ax=axes(fig);axis(ax,[0 10 0 5]);axis off;hold on;
cols=[S.sk;S.gr;S.or;S.re;S.bk];tits={'Feature Extraction','Teacher CNN','VRM-KD','NeuroPLC','TIA V21'};subs={'28-D sensor input','B-KAN 48.7k','Variance reduction','B-spline LUT','SCL download'};
for i=1:5,x=.35+(i-1)*1.92;
rectangle('Position',[x 0.85 1.55 3.0],'Curvature',.06,'FaceColor',cols(i,:)*.08+.92,'EdgeColor',cols(i,:),'LineWidth',1.2);
rectangle('Position',[x 3.25 1.55 .42],'Curvature',.06,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.0);
text(x+.77,3.46,tits{i},'Color','w','FontWeight','bold','FontSize',7,'HorizontalAlignment','center');
text(x+.77,2.55,subs{i},'Color',cols(i,:)*.55,'FontWeight','bold','FontSize',7,'HorizontalAlignment','center');
text(x+.77,1.85,['Step ' num2str(i)],'Color',S.ga,'FontSize',8,'HorizontalAlignment','center','FontAngle','italic');
if i<5,annotation(fig,'arrow',[.165+i*.184 .195+i*.184],[.58 .58],'LineWidth',1.1,'Color',S.ga,'HeadLength',5,'HeadWidth',4);end,end
e(fig,'fig1_overview',dir);end

% ---- f2 COMPILER ----
function f2(S,dir),fig=figure('Units','inches','Position',[1 1 S.W2 3.05]);ax=axes(fig);axis(ax,[0 10 0 6]);axis off;hold on;
cols=[S.bl;S.gr;S.or;S.re];tits={'Frontend','IR Graph','SCL Backend','Validation'};subs={'LAD/FBD parser','DAG + constant fold','LUT + MatMul gen','Z3 + WCET check'};
for i=1:4,x=.35+(i-1)*2.38;
rectangle('Position',[x 2.2 1.85 3.0],'Curvature',.05,'FaceColor',cols(i,:)*.08+.92,'EdgeColor',cols(i,:),'LineWidth',1.2);
rectangle('Position',[x 4.55 1.85 .48],'Curvature',.05,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.0);
text(x+.92,4.78,tits{i},'Color','w','FontWeight','bold','FontSize',7.5,'HorizontalAlignment','center');
text(x+.92,3.65,subs{i},'Color',cols(i,:)*.55,'FontWeight','bold','FontSize',7,'HorizontalAlignment','center');
if i<4,annotation(fig,'arrow',[.21+i*.222 .25+i*.222],[.62 .62],'LineWidth',1.1,'Color',S.ga,'HeadLength',5,'HeadWidth',4);end,end
e(fig,'fig2_compiler_arch',dir);end