function [xmin] = cmaes(fitness_function,N,sigma)

// INPUT PARAMETERS
// ----------------
// fitness_function -> name of the fitness function (see below for the list)
// N                -> problem dimension
// sigma            -> step size

xmin=[]; // output initialization

// --------------------  Initialization --------------------------------

xmean = rand(N,1);     // variables random initial point
// --- alternative 1: x uniform in [-5,5]
// xmean = rand(N,1)*10-5;
// --- alternative 2: x uniform in [-5,5]^2 and far from the global minimum (0,0)
// while (norm(xmean(1))<4 | norm(xmean(2))<4) do
//   xmean = rand(N,1)*10-5;
// end;

stopfitness = 1D-10;   // stop if fitness < stopfitness
stopeval = 1000*(N^2); // stop after stopeval number of function evaluations

if N==2 then           // if problem dimension is 2
  vx($+1)=xmean(1);    // save x value for 2D plot
  vy($+1)=xmean(2);    // save y value for 2D plot
end;

// Strategy parameter setting: Selection
lambda = 4+floor(3*log(N));     // population size, offspring number
// lambda = 600;                // for rastrigin fitness function only
mu = floor(lambda/2);           // number of parents/points for recombination
weights = log(mu+1)-log(1:mu)'; // muXone array for weighted recombination
weights = weights/sum(weights); // normalize recombination weights array
mueff = (sum(weights)^2)/sum(weights .^2); // variance-effective size of mu

// Strategy parameter setting: Adaptation
cc = 4/(N+4);                 // time constant for cumulation for covariance matrix
cs = (mueff+2)/((N+mueff)+3); // t-const for cumulation for sigma control
mucov = mueff;                // size of mu used for calculating learning rate ccov
ccov = (((1/mucov)*2)/((N+1.4)^2)+(1-1/mucov)*((2*mueff-1)/((N+2)^2+2*mueff))); // covariance matrix
damps = (1+2*max(0,(sqrt((mueff-1)/(N+1))-1))+cs); // damping for sigma

// Initialize dynamic (internal) strategy parameters and constants
pc = zeros(N,1);  // evolution paths for C
ps = zeros(N,1);  // evolution paths forsigma
B = eye(N,N);     // B defines the coordinate system
D = eye(N,N);     // diagonal matrix D defines the scaling
C = (B*D)*(B*D)'; // covariance matrix
chiN = (N^0.5)*(1-1/(4*N)+1/(21*(N^2))); // expectation of ||N(0,I)|| == norm(rand(N,1))

// -------------------- Generation Loop --------------------------------

counteval = 0;
while counteval<stopeval

  // Generate and evaluate lambda offspring
  z = rand(N,lambda,"normal"); // array of normally distributed mutation vectors
  for k = 1:lambda
    x(:,k) = xmean+sigma*((B*D)*z(:,k)); // add mutation
    fitness(1,k) = matrix(evstr(fitness_function+"(x(:,k))"),1,-1); // fitness function call
    counteval = counteval+1;
  end;

  // Sort by fitness and compute weighted mean into xmean
  [fitness,index] = gsort(fitness,'g','i'); // minimization
  xmean = x(:,index(1:mu))*weights;         // recombination, new mean value
  zmean = z(:,index(1:mu))*weights;         // == sigma^-1*D^-1*B''*(xmean-xold)

  // Cumulation: Update evolution paths
  ps = ((1-cs)*ps+sqrt((cs*(2-cs))*mueff)*(B*zmean));
  hsig = (norm(ps)/sqrt((1-(1-cs)^((2*counteval)/lambda))))/chiN < (1.4+2/(N+1));

  pc = ((1-cc)*pc+(hsig*sqrt((cc*(2-cc))*mueff))*((B*D)*zmean));

  // Adapt covariance matrix C
  C = (((1-ccov)*C+(ccov*(1/mucov))*(pc*pc'+(((1-hsig)*cc)*(2-cc))*C))+(((ccov*(1-1/mucov))*((B*D)*z(:,index(1:mu))))*diag(weights))*((B*D)*z(:,index(1:mu)))'); // plus rank mu update % plus rank one update% regard old matrix

  // Adapt step size sigma
  sigma = sigma*exp((cs/damps)*(norm(ps)/chiN-1));

  // Update B and D from C
  C = (triu(C)+triu(C,1)');
  [B,D] = spec(C);
  D = diag(sqrt(diag(D)));  // D contains standard deviations now

  // Break, if fitness is good enough
  if fitness(1)<=stopfitness then
    break;
  end;

  disp(string(counteval)+": "+string(fitness(1)));

  if N==2 then               // if problem dimension is 2
    vx($+1) = x(1,index(1)); // save x value for 2D plot
    vy($+1) = x(2,index(1)); // save y value for 2D plot
  end;

  // Save the best fitness and the mean fitness of this generation for plot
  best_fitness($+1) = fitness(1);
  mean_fitness($+1) = mean(fitness);
end;

// -------------------- Ending Message ---------------------------------
disp(string(counteval)+": "+string(fitness(1)));
xmin = x(:,index(1)); // Return best point of last generation.

if N==2 then                // if problem dimension is 2

  // --- 3D plot of the function
  scf();                    // new plot
  x=[-5:.1:5];              // range
  if (fitness_function=="frast" | fitness_function=="frast10" | fitness_function=="frast1000") then
    subplot(211);           // plot in the upper half of the figure
  end;
  z=eval3d(eval(fitness_function+"_2"),x); // z=f(x)
  plot3d1(x,x,z);
  xset("fpf"," ")           // don't plot level curves' values

  // --- 2D plot of the level curves
  if (fitness_function=="frast" | fitness_function=="frast10" | fitness_function=="frast1000") then
    subplot(212);           // plot in the lower half of the figure
    contour2d(x,x,z,10);
  else
    contour(x,x,z,10,flag=[1 2 4], zlev=-10); // plot level curves on plan z=-10
  end;
  f=get("current_figure");
  f.figure_size=[800,600];  // change figure size

  // --- 2D plot of the level curves and the path of the solution
  scf();
  contour2d(x,x,z,10);
  plot2d(vx,vy);           // plot the path taken by the solution
  f=get("current_figure");
  f.figure_size=[800,600];
end;

// --- plot the best fitness
scf();
plot2d(best_fitness);
f=get("current_figure");
f.figure_size=[800,600];
a=f.children;
a.log_flags="nln";         // logarithmic scale of the y axe
a.title.text=fitness_function+": fitness migliore (in rosso) e fitness media (in nero) in funzione della generazione"
a.x_label.text="generazione";
a.y_label.text="fitness";
p=f.children.children.children;
p.line_mode="off";         // don't plot line between points
p.mark_mode="on";          // plot marks
p.mark_size_unit="point";
p.mark_size=3;
p.mark_foreground=5;

// --- plot the mean fitness
plot2d(mean_fitness);
f=get("current_figure");
p=f.children.children.children;
p.line_mode="off";
p.mark_mode="on";
p.mark_size_unit="point";
p.mark_size=3;

endfunction

// =================== Fitness functions ===============================

// --- n-dimensional fitness functions

function [f] = fsphere(x)
  f = sum(x .^2,'r');
endfunction

function [f] = fschwefel(x)
  f = 0;
  for i = 1:size(x,1)
    f = f+sum(x(1:i))^2;
  end;
endfunction

function [f] = fcigar(x)
  f = (x(1))^2+1000000*sum(x(2:$).^2,'r');
endfunction

function [f] = fcigtab(x)
  f = ((x(1))^2+100000000*(x($)^2))+10000*sum(x(2:$-1).^2,'r');
endfunction

function [f] = ftablet(x)
  f = 1000000*((x(1))^2)+sum(x(2:$).^2,'r');
endfunction

function [f] = felli(x)
  N = size(x,1);
  if N<2 then
    error("dimension must be greater one");
  end;
  f = (1000000 .^((0:N-1)/(N-1)))*(x .^2);
endfunction

function [f] = felli100(x)
  N = size(x,1);
  if N<2 then
    error("dimension must be greater one");
  end;
  f = (10000 .^((0:N-1)/(N-1)))*(x .^2);
endfunction

function [f] = fdiffpow(x)
  N = size(x,1);
  if N<2 then
    error("dimension must be greater one");
  end;
  f = sum(abs(x) .^(2+(10*(0:N-1)')/(N-1)),'r');
endfunction

function [f] = frastrigin10(x)
  N = size(x,1);
  if N<2 then
    error("dimension must be greater one");
  end;
  scale = 10 .^((0:N-1)'/(N-1));
  f = 10*size(x,1)+sum(((scale .*x) .^2-10*cos((2*%pi)*(scale .*x))),'r');
endfunction

function [f] = frastn(x,n)
  N = size(x,1);
  if N<2 then
    error("dimension must be greater one");
  end;
  scale = n .^((0:N-1)'/(N-1));
  f = 10*size(x,1)+sum(((scale .*x) .^2-10*cos((2*%pi)*(scale .*x))),'r');
endfunction

function [f] = frast(x)
   f = frastn(x,1)
endfunction

function [f] = frast10(x)
   f = frastn(x,10)
endfunction

function [f] = frast1000(x)
   f = frastn(x,1000)
endfunction

// --- 2-dimensional fitness functions

function [f] = fsphere_2(x,y)
  f = x^2+y^2;
endfunction

function [f] = fschwefel_2(x,y)
  f = x^2 + (x+y)^2;
endfunction

function [f] = fcigar_2(x,y)
  f = x^2 + 1000000*y^2;
endfunction

function [f] = fdiffpow_2(x,y)
  f = abs(x)^2 + abs(y)^12;
endfunction

function [f]= frastn_2(x,y,n)
  f=20+(x^2-10*cos(2*%pi*x)+y^2-10*cos(2*%pi*n*y));
endfunction

function [f] = frast_2(x,y)
  f=frastn_2(x,y,1);
endfunction

function [f] = frast10_2(x,y)
  f=frastn_2(x,y,10);
endfunction

function [f] = frast1000_2(x,y)
  f=frastn_2(x,y,1000);
endfunction
