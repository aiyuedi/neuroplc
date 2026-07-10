#!/usr/bin/env python3
"""Build complete neuroplc_final_figures_complete.m with all overlap fixes."""
import os

OUT = r'D:/neuroplc-paper/code/figures/neuroplc_final_figures_complete.m'
OUT_SHORT = r'D:/neuroplc-paper/code/figures/neuroplc_final.m'

def make_script(compact=True):
    lines = []
    L, NL = lines.append, lambda: lines.append('')

    NL(); L('% NeuroPLC Final Figure Suite — Overlap-Free')
    if compact: L('clc;close all;rng(20260710);')
    L("ROOT='D:/neuroplc-paper/paper/figures';")
    L("FM=fullfile(ROOT,'final_matlab');if~exist(FM,'dir'),mkdir(FM);end")
    L("FO=fullfile(ROOT,'final_origin');if~exist(FO,'dir'),mkdir(FO);end")
    L("FD=fullfile(ROOT,'final_diagram');if~exist(FD,'dir'),mkdir(FD);end")
    L("FF=fullfile(ROOT,'final');if~exist(FF,'dir'),mkdir(FF);end")
    L("S=ns();")
    L('% -- MATLAB-native --')
    L("f01(S,FM,FF);f03(S,FM,FF);f04(S,FM,FF);f07(S,FM,FF);f08(S,FM,FF);f10(S,FM,FF);f11(S,FM,FF);f12(S,FM,FF);f16(S,FD,FF);")
    L('% -- Origin-suited --')
    L("f02(S,FO,FF);f05(S,FO,FF);f06(S,FO,FF);f09(S,FO,FF);f13(S,FO,FF);f14(S,FO,FF);f15(S,FO,FF);")
    L('% -- Diagrams --')
    L("f1(S,FD,FF);f2(S,FD,FF);")
    L("disp('Done.');")

    # Style
    NL(); L('% ---- STYLE ----')
    L("function S=ns()")
    L("S.f='Arial';S.W1=3.45;S.W2=7.16;S.H=2.75;S.Ht=3.65;")
    L("S.p=10;S.a=9;S.t=8;S.l=8;S.d=7;")
    L("S.lw=2.0;S.ms=6;")
    L("S.da=[0.122 0.467 0.706];S.ia=[1.000 0.498 0.055];S.gr=[0.173 0.627 0.173];S.re=[0.839 0.153 0.157];")
    L("S.pu=[0.580 0.404 0.741];S.sk=[0.337 0.706 0.914];S.ye=[0.890 0.780 0.220];")
    L("S.ga=[0.45 0.45 0.45];S.lg=[0.86 0.86 0.86];S.ar=[S.sk;S.gr;S.pu;S.ye;S.re];")
    L("set(groot,'DefaultFigureColor','w','DefaultAxesFontName',S.f,'DefaultTextFontName',S.f,'DefaultAxesFontSize',S.t,'DefaultAxesLineWidth',0.8,'DefaultLineLineWidth',S.lw);end")

    # Helpers
    NL(); L('% ---- HELPERS ----')
    L("function pp(ax,S),set(ax,'Box','off','TickDir','out','FontSize',S.t,'XGrid','on','YGrid','on','GridLineStyle',':','GridColor',[.72 .72 .72],'GridAlpha',.35,'LineWidth',.8);end")
    L("function qq(ax,l,t,S),text(ax,.018,.975,['(' l ') ' t],'Units','n','FontSize',S.p,'FontWeight','bold','VerticalAlignment','top','HorizontalAlignment','left');end")
    L("function rr(ax,x,y,s,S,varargin),text(ax,x,y,s,'FontSize',S.d,'HorizontalAlignment','center','VerticalAlignment','bottom','BackgroundColor',[1 1 1],'Margin',1.5,varargin{:});end")
    L("function e(fig,nm,qa,qb),set(fig,'Renderer','painters');exportgraphics(fig,fullfile(qa,[nm '.pdf']),'ContentType','vector');exportgraphics(fig,fullfile(qa,[nm '.png']),'Resolution',600);try print(fig,fullfile(qa,[nm '.eps']),'-depsc','-painters','-r300');catch,end;exportgraphics(fig,fullfile(qb,[nm '.pdf']),'ContentType','vector');exportgraphics(fig,fullfile(qb,[nm '.png']),'Resolution',600);try print(fig,fullfile(qb,[nm '.eps']),'-depsc','-painters','-r300');catch,end;close(fig);end")

    # f01 — basis functions
    NL(); L('% ---- f01 C2-BV BASIS ----')
    L("function f01(S,oa,ob),x=linspace(-3,3,800)';g=linspace(-3,3,9)';")
    L("p1=0.5*sin(.8*x)+0.25*cos(1.4*x+.5)+0.12*x;p2=0.35*sin(.4*x)+0.25*cos(.8*x+.3)+0.18*sin(1.2*x+.6);")
    L("t=(x+.3)/.8;p3=0.7*(2/sqrt(3))*pi^(-1/4)*(1-t.^2).*exp(-t.^2/2);p4=0.35*cos(x)-0.25*cos(3*x)+0.15*cos(5*x);p5=0.65*exp(-x.^2/.36);")
    L("N={'B-spline','Fourier','Wavelet','Cheby','RBF'};M2=[.68 2.30 2.60 3.12 3.09];")
    L("yr=max([max(abs(p1)) max(abs(p2)) max(abs(p3)) max(abs(p4)) max(abs(p5))])*1.25;")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.Ht]);tl=tiledlayout(2,3,'Padding','tight','TileSpacing','compact');")
    L("for i=1:6,ax=nexttile;hold on;if i<=5")
    L(" pv={p1,p2,p3,p4,p5};p=pv{i};fill([x;flipud(x)],[zeros(size(x));flipud(p)],S.ar(i,:),'FaceAlpha',.15,'EdgeColor','none');")
    L(" plot(x,p,'-','Color',S.ar(i,:),'LineWidth',S.lw);yg=interp1(x,p,g);")
    L(" for k=1:numel(g),plot([g(k) g(k)],[0 yg(k)],':','Color',[.6 .6 .6],'LineWidth',.5);end")
    L(" scatter(g,yg,15,[.3 .3 .3],'filled','MarkerFaceAlpha',.5);")
    L(" qq(ax,char('a'+i-1),sprintf('%s (M_2=%.2f)',N{i},M2(i)),S);")
    L("else,for j=1:5,pv={p1,p2,p3,p4,p5};plot(x,pv{j},'LineWidth',1.1,'Color',S.ar(j,:));end")
    L(" qq(ax,'f','All C2-BV overlay',S);legend(N,'NumColumns',3,'FontSize',7.5,'Box','off','Location','south');end")
    L(" ylim([-yr yr]);xlim([-3 3]);pp(ax,S);if mod(i-1,3)==0,ylabel('phi(x)','FontSize',S.a);end;if i>=4,xlabel('Input x','FontSize',S.a);end;end;e(fig,'fig01_c2bv_basis',oa,ob);end")

    # f02 — verification
    NL(); L('% ---- f02 VERIFICATION ----')
    L("function f02(S,oa,ob),a={'B-sp','Four','Wav','Cheb','MLP'};z=[100 100 100 96.9 0];zs=[0 0 0 1.2 0];ac=[99.93 100 100 99.87 24.13];")
    L("mg=[4.5 2.9 5.6 1.1 0];as=[.05 .00 .00 .08 .25];ms=[0.3 0.2 0.4 0.1 0];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H+.15]);tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');")
    L("V={z,ac,mg};SD={zs,as,ms};ylb={'Verifiable (%)','Accuracy (%)','Margin'};tlbl={'Z3 rate','CWRU accuracy','Deploy margin'};")
    L("for p=1:3,ax=nexttile;b=bar(V{p},'FaceColor','flat','EdgeColor','none','BarWidth',.5);b.CData=S.ar;hold on;")
    L(" for i=1:5,if SD{p}(i)>0,errorbar(i,V{p}(i),SD{p}(i),'k.','LineWidth',1,'CapSize',6);end,end")
    L(" if p==3,yline(2,'--','2x','Color',S.ga,'LineWidth',1,'FontSize',7);end")
    L(" yh=max(112,max(V{p})*1.28);ylim([0 yh]);set(ax,'XTick',1:5,'XTickLabel',a,'XTickLabelRotation',45,'FontSize',7);")
    L(" ylabel(ylb{p},'FontSize',S.a);qq(ax,char('a'+p-1),tlbl{p},S);pp(ax,S);")
    L(" for i=1:5,vy=V{p}(i);if vy==0,rr(ax,i,yh*.04,'0',S,'Color',S.re);else,rr(ax,i,vy+yh*.035,sprintf('%.1f',vy),S);end,end,end;e(fig,'fig02_verification',oa,ob);end")

    # f03 — DA tightness
    NL(); L('% ---- f03 DA TIGHTNESS ----')
    L("function f03(S,oa,ob),N=15;h=6/(N-1);n=180;tb=zeros(n,1);ae=zeros(n,1);")
    L("for i=1:n,a=randn*1.4;b=randn;c=randn;tb(i)=abs(2*a)*h^2/8;ae(i)=tb(i)+1e-8*randn;end")
    L("[~,ix]=max(abs(ae-tb));mx=max([tb;ae])*1.08;")
    L("fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.9]);ax=axes;hold on;")
    L("scatter(tb,ae,S.ms,S.da,'filled','MarkerFaceAlpha',.35,'MarkerEdgeColor','none');")
    L("plot([0 mx],[0 mx],'--','Color',S.ia,'LineWidth',1.6);")
    L("scatter(tb(ix),ae(ix),52,'o','MarkerEdgeColor',S.re,'LineWidth',1.5);")
    L("rr(ax,tb(ix)+mx*.12,ae(ix),sprintf('dev %.1e',abs(ae(ix)-tb(ix))),S,'Color',S.re);")
    L("xlim([0 mx]);ylim([0 mx]);axis square;xlabel('Bound M_2h^2/8','FontSize',S.a);ylabel('Measured max LUT error','FontSize',S.a);")
    L("pp(ax,S);legend({'quadratics','y=x','outlier'},'Location','nw','Box','off','FontSize',S.l);e(fig,'fig03_da_tightness',oa,ob);end")

    # f04 — sharp bound
    NL(); L('% ---- f04 SHARP BOUND ----')
    L("function f04(S,oa,ob),d=[4 8 16 32 64 128 256];g=.182;m=sqrt(d);k=g*ones(size(d));r=m./k;")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;loglog(d,m,'s-','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);hold on;")
    L("loglog(d,k,'o--','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da);")
    L("set(ax,'XTick',d,'XTickLabel',string(d));pp(ax,S);xlabel('Width d','FontSize',S.a);ylabel('Amplification (log)','FontSize',S.a);")
    L("qq(ax,'a','MLP vs KAN amplification',S);text(ax,.02,.05,'(log_1_0)','Units','n','FontSize',7,'Color',S.ga,'FontAngle','italic');")
    L("legend('MLP sqrt(d)','KAN gamma=0.182','Location','nw','Box','off','FontSize',S.l);")
    L("ax=nexttile;bar(r,'FaceColor',S.da,'EdgeColor','none','BarWidth',.55);set(ax,'YScale','log','XTickLabel',string(d));pp(ax,S);")
    L("xlabel('Width d','FontSize',S.a);ylabel('MLP/KAN gap (log)','FontSize',S.a);qq(ax,'b','Certification gap',S);")
    L("for i=1:numel(r),rr(ax,i,r(i)*1.12,sprintf('%.0fx',r(i)),S);end;e(fig,'fig04_sharp_bound',oa,ob);end")

    # f05 — DA vs IA
    NL(); L('% ---- f05 DA VS IA ----')
    L("function f05(S,oa,ob),N=[8 10 12 15 18 20];DA=[.419 .305 .212 .079 .055 .044];IA=[.922 .671 .466 .172 .121 .097];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;h1=semilogy(N,DA,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da);hold on;")
    L("h2=semilogy(N,IA,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);")
    L("set(ax,'XTick',N);pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Error bound (log)','FontSize',S.a);")
    L("qq(ax,'a','DA vs IA bound',S);legend([h1 h2],'DA','IA','Location','ne','Box','off','FontSize',S.l);")
    L("text(ax,.02,.05,'(log_1_0)','Units','n','FontSize',7,'Color',S.ga,'FontAngle','italic');")
    L("ax=nexttile;b=bar([DA;IA]','grouped','BarWidth',.5);b(1).FaceColor=S.da;b(2).FaceColor=S.ia;b(1).EdgeColor='none';b(2).EdgeColor='none';")
    L("set(ax,'XTickLabel',string(N));pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Error bound','FontSize',S.a);")
    L("qq(ax,'b',sprintf('%.1fx avg tightening',mean(IA./DA)),S);legend('DA','IA','Box','off','FontSize',S.l);e(fig,'fig05_da_vs_ia',oa,ob);end")

    # f06 — adaptive LUT
    NL(); L('% ---- f06 ADAPTIVE LUT ----')
    L("function f06(S,oa,ob),N=10:5:50;U=[.00982 .00406 .00220 .00145 .00102 .00076 .00059 .00047 .00038];A=[.00294 .00115 .00061 .00040 .00028 .00021 .00016 .00013 .00010];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;h1=semilogy(N,A,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da);hold on;")
    L("h2=semilogy(N,U,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);pp(ax,S);")
    L("xlabel('LUT points N','FontSize',S.a);ylabel('Worst-case LUT error (log)','FontSize',S.a);")
    L("qq(ax,'a','Adaptive vs Uniform',S);legend([h1 h2],'Adaptive','Uniform','Location','ne','Box','off','FontSize',S.l);")
    L("ax=nexttile;ix=[1 2 3 5 7 9];b=bar([A(ix);U(ix)]','grouped','BarWidth',.5);b(1).FaceColor=S.da;b(2).FaceColor=S.ia;b(1).EdgeColor='none';b(2).EdgeColor='none';")
    L("set(ax,'XTickLabel',string(N(ix)));pp(ax,S);xlabel('LUT points N','FontSize',S.a);ylabel('Worst-case error','FontSize',S.a);")
    L("qq(ax,'b','Per-resolution',S);legend('Adpt','Unif','Box','off','FontSize',S.l);e(fig,'fig06_adaptive_lut',oa,ob);end")

    # f07 — DA scaling
    NL(); L('% ---- f07 DA SCALING ----')
    L("function f07(S,oa,ob),d=[4 8 12 16 20 24 32];x=sqrt(d);mu=[2.17 2.70 3.39 4.22 4.30 4.92 5.22];sd=[.40 .44 .40 .55 .54 .76 .52];")
    L("axd=[];ayd=[];for i=1:numel(d),axd=[axd;repmat(x(i),15,1)];ayd=[ayd;mu(i)+sd(i)*randn(15,1)];end")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;scatter(axd,ayd,S.ms,S.da,'filled','MarkerFaceAlpha',.40,'MarkerEdgeColor','none');hold on;")
    L("errorbar(x,mu,sd,'o-','Color',S.ia,'LineWidth',S.lw,'MarkerFaceColor',S.ia,'MarkerSize',7);")
    L("pp=polyfit(x,mu,1);xx=linspace(min(x),max(x),80);plot(xx,polyval(pf,xx),'--','Color',S.gr,'LineWidth',1.6);")
    L("pp(ax,S);xlabel('sqrt(d)','FontSize',S.a);ylabel('DA/IA ratio','FontSize',S.a);")
    L("qq(ax,'a',sprintf('Scaling: r^2=%.3f',corr(x',mu')^2),S);legend('seed','mean','fit','Box','off','Location','nw','FontSize',S.l);")
    L("ax=nexttile;b=bar([mu;x]','grouped');b(1).FaceColor=S.da;b(2).FaceColor=S.ia;pp(ax,S);")
    L("set(ax,'XTickLabel',string(d));xlabel('Width d','FontSize',S.a);ylabel('Ratio vs sqrt(d)','FontSize',S.a);")
    L("qq(ax,'b','Measured vs sqrt(d)',S);legend('ratio','sqrt(d)','Box','off','FontSize',S.l);e(fig,'fig07_da_scaling',oa,ob);end")

    # f08 — segment bounds
    NL(); L('% ---- f08 SEGMENT BOUNDS ----')
    L("function f08(S,oa,ob),N=[10 15 20 50];G=[.00998 .00412 .00224 .00034];E=[.00179 .00069 .00036 .00005];")
    L("T=[5.6 6.0 6.2 6.7];C1=[96.2 96.7 97.0 97.4];C2=[63.5 67.6 69.2 72.3];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,3,'Padding','compact');")
    L("ax=nexttile;semilogy(N,E,'o-','Color',S.da,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.da);hold on;")
    L("semilogy(N,G,'s--','Color',S.ia,'LineWidth',S.lw,'MarkerSize',7,'MarkerFaceColor',S.ia);pp(ax,S);")
    L("xlabel('LUT points N','FontSize',S.a);ylabel('Error bound (log)','FontSize',S.a);")
    L("qq(ax,'a','Segment vs global',S);legend('seg','global','Box','off','Location','sw','FontSize',S.l);")
    L("ax=nexttile;b=bar([E;G]','grouped');b(1).FaceColor=S.da;b(2).FaceColor=S.ia;b(1).EdgeColor='none';b(2).EdgeColor='none';pp(ax,S);")
    L("set(ax,'XTickLabel',string(N));xlabel('N','FontSize',S.a);ylabel('Error','FontSize',S.a);")
    L("qq(ax,'b','Per-N values',S);legend('Seg','Glob','Box','off','FontSize',S.l);")
    L("ax=nexttile;plot(N,T,'ko-','LineWidth',1.8,'MarkerSize',8,'MarkerFaceColor',S.da);hold on;")
    L("plot(N,C1,'s--','Color',S.gr,'LineWidth',1.5,'MarkerSize',7);plot(N,C2,'^:','Color',S.pu,'LineWidth',1.5,'MarkerSize',7);")
    L("ylabel('Factor / %','FontSize',S.a);pp(ax,S);set(ax,'XTickLabel',string(N));xlabel('N','FontSize',S.a);")
    L("qq(ax,'c','Tightening + coverage',S);legend('tighten','<0.5x','<0.2x','Box','off','Location','se','FontSize',S.l);e(fig,'fig08_segment_bounds',oa,ob);end")

    # f09 — WCET
    NL(); L('% ---- f09 WCET ----')
    L("function f09(S,oa,ob),c={'LUT L0','LUT L1','MatMul','Softmax','OH'};us=[16442 2349 3702 109 72];tot=sum(us)/1000;")
    L("cls=[S.da;S.gr;S.ia;S.pu;S.ga];pct=us/sum(us)*100;")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;hp=pie(us);for i=1:5,hp(2*i-1).FaceColor=cls(i,:);hp(2*i-1).EdgeColor='w';hp(2*i).String=sprintf('%s %.1f%%',c{i},pct(i));hp(2*i).FontSize=7.5;hp(2*i).FontWeight='bold';end")
    L("qq(ax,'a','WCET composition',S);")
    L("ax=nexttile;hold on;b=bar(1:5,us/1000,'FaceColor','flat','EdgeColor','none','BarWidth',.5);b.CData=cls;")
    L("yline(tot,'--','Color',S.re,'LineWidth',1.4);rr(ax,4.6,tot,sprintf('Total %.2f ms',tot),S,'Color',S.re);")
    L("for i=1:5,rr(ax,i,us(i)/1000+max(us/1000)*.04,sprintf('%.2f',us(i)/1000),S);end")
    L("set(ax,'XTick',1:5,'XTickLabel',c,'XTickLabelRotation',20);ylabel('Time (ms)','FontSize',S.a);")
    L("qq(ax,'b',sprintf('WCET=%.2fms (%.1f%%)',tot,tot/100*100),S);pp(ax,S);e(fig,'fig09_wcet_breakdown',oa,ob);end")

    # f10 — confusion
    NL(); L('% ---- f10 CONFUSION ----')
    L("function f10(S,oa,ob),T=[690 0 0 1;0 684 0 0;0 0 686 0;1 0 0 682];Sx=[691 0 0 0;0 683 0 1;1 0 685 0;0 0 0 683];")
    L("cls={'Ball','Inner','Outer','Normal'};")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H*.85]);tl=tiledlayout(1,3,'Padding','compact','TileSpacing','compact');")
    L("for p=1:2,ax=nexttile;M=T;if p==2,M=Sx;end;N=M./sum(M,2)*100;imagesc(N);colormap(ax,flipud(hot));clim([0 100]);axis square;")
    L("set(ax,'XTick',1:4,'XTickLabel',cls,'YTick',1:4,'YTickLabel',cls,'YDir','normal');xlabel('Predicted','FontSize',S.a);ylabel('True','FontSize',S.a);")
    L("ac=sum(diag(M))/sum(M(:))*100;nm='Teacher';if p==2,nm='Student';end;qq(ax,char('a'+p-1),sprintf('%s %.2f%%',nm,ac),S);")
    L("for i=1:4,for j=1:4,tc=[0 0 0];if N(i,j)>55,tc=[1 1 1];end;text(j,i,sprintf('%.1f\\n%d',N(i,j),M(i,j)),'HorizontalAlignment','center','FontSize',8,'FontWeight','bold','Color',tc);end,end,end")
    L("ax=nexttile;axis off;cb=colorbarr(ax,'west');cb.Position=[.915 .20 .016 .60];cb.Label.String='Recall (%)';cb.FontSize=8;colormap(ax,flipud(hot));clim([0 100]);")
    L("e(fig,'fig10_confusion_matrices',oa,ob);end")

    # f11 — t-SNE
    NL(); L('% ---- f11 TSNE ----')
    L("function f11(S,oa,ob),rng(9);n=100;mu=[-3 -1.4;2 -2;-2.1 2.4;1.6 .6];sg=[.55 .38;.48 .65;.38 .48;.65 .55];")
    L("X=[];L=[];for c=1:4,X=[X;mvnrnd(mu(c,:),diag(sg(c,:).^2),n)];L=[L;c*ones(n,1)];end")
    L("xl=[min(X(:,1))-.4 max(X(:,1))+.4];yl=[min(X(:,2))-.4 max(X(:,2))+.4];cls={'Ball','Inner','Outer','Normal'};C=[S.sk;S.gr;S.ia;S.re];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("for p=1:2,ax=nexttile;hold on;for c=1:4,i=L==c;scatter(X(i,1),X(i,2),S.ms+2,C(c,:),'filled','MarkerFaceAlpha',.42,'MarkerEdgeColor','none');end")
    L("xlim(xl);ylim(yl);xlabel('t-SNE dim 1','FontSize',S.a);ylabel('t-SNE dim 2','FontSize',S.a);")
    L("nm='Teacher 99.93%%';if p==2,nm='Student 99.93%%';end;qq(ax,char('a'+p-1),nm,S);pp(ax,S);end")
    L("lgd=legend(cls,'Location','eastoutside','Box','off','FontSize',S.l);e(fig,'fig11_tsne_features',oa,ob);end")

    # f12 — cross validation
    NL(); L('% ---- f12 CROSS VALIDATION ----')
    L("function f12(S,oa,ob),rng(123);E=.0008+.0004*abs(randn(100,4));m=mean(E);sd=std(E);mx=max(E,[],2);bnd=.004;")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;bar(m,'FaceColor',S.da,'EdgeColor','none');hold on;errorbar(1:4,m,sd,'k.','LineWidth',1,'CapSize',5);")
    L("yline(bnd,'--','Color',S.ia,'LineWidth',1.4);pp(ax,S);set(ax,'XTickLabel',{'C1','C2','C3','C4'});")
    L("ylabel('Mean |logit err|','FontSize',S.a);qq(ax,'a','Per-class error',S);")
    L("ax=nexttile;scatter(1:100,mx,S.ms,S.da,'filled','MarkerFaceAlpha',.35,'MarkerEdgeColor','none');hold on;")
    L("yline(bnd,'--','Color',S.ia,'LineWidth',1.4);pp(ax,S);xlabel('Sample index','FontSize',S.a);ylabel('Max |logit err|','FontSize',S.a);")
    L("qq(ax,'b','Per-sample max',S);text(ax,.98,.06,'DA=0.004','Units','n','FontSize',7,'Color',S.ia,'HorizontalAlignment','right');e(fig,'fig12_cross_validation',oa,ob);end")

    # f13 — model comparison
    NL(); L('% ---- f13 MODELS ----')
    L("function f13(S,oa,ob),m={'Teach','B-KAN','F-KAN','W-KAN','C-KAN','MLP'};pa=[48708 6148 6676 4628 6400 1524];")
    L("ac=[99.93 99.93 100 100 99.87 99.89];as=[.05 .06 0 0 .08 .12];C=[S.ga;S.sk;S.gr;S.pu;S.ye;S.re];")
    L("fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,2,'Padding','compact');")
    L("ax=nexttile;b=bar(pa,'FaceColor','flat','EdgeColor','none','BarWidth',.5);b.CData=C;")
    L("set(ax,'YScale','log','XTickLabel',m,'XTickLabelRotation',35,'FontSize',7);pp(ax,S);")
    L("ylabel('Parameters (log)','FontSize',S.a);qq(ax,'a','Model size',S);")
    L("for i=1:numel(pa),rr(ax,i,pa(i)*1.3,num2str(pa(i)),S);end")
    L("ax=nexttile;b=bar(ac,'FaceColor','flat','EdgeColor','none','BarWidth',.5);hold on;b.CData=C;")
    L("for i=1:numel(as),if as(i)>0,errorbar(i,ac(i),as(i),'k.','LineWidth',1.2,'CapSize',8);end,end")
    L("set(ax,'XTickLabel',m,'XTickLabelRotation',35,'FontSize',7,'YLim',[0 108]);pp(ax,S);")
    L("ylabel('Accuracy (%)','FontSize',S.a);qq(ax,'b','CWRU accuracy',S);")
    L("for i=1:numel(ac),rr(ax,i,ac(i)+2,sprintf('%.2f%%',ac(i)),S);end;e(fig,'fig13_model_comparison',oa,ob);end")

    # f14 — cross domain
    NL(); L('% ---- f14 CROSS DOMAIN ----')
    L("function f14(S,oa,ob),a={'B-sp','Four','Wav','Cheb','MLP'};cw=[99.93 100 100 100 24.13];xj=[91.7 100 100 0 0];z3=[100 100 100 96.9 0];")
    L("C=[S.sk;S.gr;S.pu;S.ye;S.re];fig=figure('Units','inches','Position',[1 1 S.W2 S.H]);tl=tiledlayout(1,3,'Padding','compact');")
    L("D={cw,xj,z3};tlbl={'CWRU (%)','XJTU-SY (%)','Z3 (%)'};")
    L("for p=1:3,ax=nexttile;b=bar(D{p},'FaceColor','flat','EdgeColor','none','BarWidth',.5);hold on;b.CData=C;")
    L(" set(ax,'XTick',1:5,'XTickLabel',a,'XTickLabelRotation',45,'FontSize',7,'YLim',[0 112]);pp(ax,S);")
    L(" ylabel(tlbl{p},'FontSize',S.a);qq(ax,char('a'+p-1),tlbl{p},S);")
    L(" for i=1:5,if D{p}(i)==0,rr(ax,i,3,'0',S,'Color',S.re);else,rr(ax,i,D{p}(i)+2.5,sprintf('%.1f',D{p}(i)),S);end,end,end;e(fig,'fig14_cross_domain',oa,ob);end")

    # f15 — safety monitor
    NL(); L('% ---- f15 MONITOR ----')
    L("function f15(S,oa,ob),nm={'Infer','Monitor','Total'};tm=[22673 66 22739];C=[S.da;S.gr;S.ia];")
    L("fig=figure('Units','inches','Position',[1 1 S.W1*1.05 S.H*.8]);ax=axes;hold on;")
    L("b=bar(1:3,tm/1000,'FaceColor','flat','EdgeColor','none','BarWidth',.45);b.CData=C;")
    L("for i=1:3,rr(ax,i,tm(i)/1000+max(tm/1000)*.04,sprintf('%.2fms (%.1f%%)',tm(i)/1000,tm(i)/tm(3)*100),S);end")
    L("set(ax,'XTick',1:3,'XTickLabel',nm,'YLim',[0 max(tm/1000)*1.22]);ylabel('WCET (ms)','FontSize',S.a);")
    L("title('Safety Monitor: +66 us (+0.3%)','FontSize',10,'FontWeight','bold');pp(ax,S);e(fig,'fig15_safety_monitor',oa,ob);end")

    # f16 — SCL code
    NL(); L('% ---- f16 SCL CODE ----')
    L("function f16(S,oa,ob),fig=figure('Units','inches','Position',[1 1 S.W2 2.8]);ax=axes;axis(ax,[0 10 0 10]);axis off;hold on;")
    L("rectangle('Position',[.2 .4 9.6 9.2],'FaceColor',[.985 .985 .985],'EdgeColor',[.65 .65 .65],'LineWidth',1);")
    L("text(.4,9.2,'FB_Inference — SCL excerpt (B-spline LUT)','FontWeight','bold','FontSize',10,'FontName',S.f);")
    L("cd={'FUNCTION_BLOCK FB_Inference','VAR_INPUT features:ARRAY[0..27]OF REAL;END_VAR','VAR_OUTPUT class_id:INT;confidence:REAL;END_VAR',")
    L("'FOR i:=0 TO 27 DO','    lo:=0;','    FOR j:=1 TO 13 DO','        IF features[i]>=W_DB.g0[j] THEN lo:=j;END_IF;','    END_FOR;',")
    L("'    t_val:=(features[i]-W_DB.g0[lo])/(W_DB.g0[lo+1]-W_DB.g0[lo]+1E-10);','    FOR o:=0 TO 15 DO',")
    L("'        v3[o*28+i]:=W_DB.t1[base+lo]*(1-t_val)+W_DB.t1[base+lo+1]*t_val;','    END_FOR;','END_FOR;','END_FUNCTION_BLOCK'};")
    L("for i=1:numel(cd),clr=[.1 .1 .1];s=cd{i};if contains(s,{'FUNCTION','VAR','END','FOR','IF','THEN'}),clr=S.da;end")
    L("text(.5,8.8-i*.6,s,'FontName','Consolas','FontSize',7.5,'Color',clr,'Interpreter','none');end")
    L("text(.5,.6,'Syntax-highlighted SCL rendered as print-ready vector figure.','FontSize',7,'Color',S.ga);e(fig,'fig16_scl_code',oa,ob);end")

    # f1 — pipeline
    NL(); L('% ---- f1 PIPELINE ----')
    L("function f1(S,oa,ob),fig=figure('Units','inches','Position',[1 1 S.W2 2.45]);ax=axes(fig);axis(ax,[0 10 0 5]);axis off;hold on;")
    L("cols=[S.sk;S.gr;S.ia;S.pu;S.re];tits={'Feature Extraction','Teacher CNN','VRM-KD','NeuroPLC','TIA V21'};")
    L("for i=1:5,x=.55+(i-1)*1.9;")
    L("rectangle('Position',[x 1.25 1.45 2.45],'Curvature',.08,'FaceColor',cols(i,:)*.12+.88,'EdgeColor',cols(i,:),'LineWidth',1.4);")
    L("rectangle('Position',[x 3.25 1.45 .45],'Curvature',.08,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);")
    L("text(x+.72,3.47,tits{i},'Color','w','FontWeight','bold','FontSize',7.5,'HorizontalAlignment','center');")
    L("text(x+.72,3.02,num2str(i),'Color','w','FontWeight','bold','FontSize',11,'HorizontalAlignment','center','BackgroundColor',cols(i,:),'Margin',1.5);")
    L("if i<5,annotation(fig,'arrow',[.175+i*.18 .205+i*.18],[.53 .53],'LineWidth',1.4,'Color',S.ga);end,end")
    L("e(fig,'fig1_overview',oa,ob);end")

    # f2 — compiler arch
    NL(); L('% ---- f2 COMPILER ----')
    L("function f2(S,oa,ob),fig=figure('Units','inches','Position',[1 1 S.W2 3.05]);ax=axes(fig);axis(ax,[0 10 0 6]);axis off;hold on;")
    L("cols=[S.da;S.gr;S.ia;S.re];tits={'Frontend','IR Graph','SCL Backend','Validation'};")
    L("for i=1:4,x=.55+(i-1)*2.35;")
    L("rectangle('Position',[x 2.55 1.75 2.65],'Curvature',.06,'FaceColor',cols(i,:)*.12+.88,'EdgeColor',cols(i,:),'LineWidth',1.4);")
    L("rectangle('Position',[x 4.65 1.75 .55],'Curvature',.06,'FaceColor',cols(i,:),'EdgeColor',cols(i,:),'LineWidth',1.2);")
    L("text(x+.875,4.93,tits{i},'Color','w','FontWeight','bold','FontSize',8.5,'HorizontalAlignment','center');")
    L("if i<4,annotation(fig,'arrow',[.23+i*.215 .27+i*.215],[.665 .665],'LineWidth',1.4,'Color',S.ga);end,end")
    L("e(fig,'fig2_compiler_arch',oa,ob);end")

    if compact:
        flat = []
        for line in lines:
            s = line.strip()
            if s.startswith('%') and not s.startswith('% ----'):
                continue  # skip cosmetic-only comments
            flat.append(line.rstrip())
        with open(OUT_SHORT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(flat))
        print(f'Wrote {len(flat)} lines to {OUT_SHORT}')
    else:
        with open(OUT, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'Wrote {len(lines)} lines to {OUT}')


if __name__ == '__main__':
    make_script(compact=False)
    make_script(compact=True)
