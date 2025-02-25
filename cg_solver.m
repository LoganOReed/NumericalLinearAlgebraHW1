function resvec = cg_solver(alpha, beta, N, max_iter)

    h = 1 / (N + 1); % Grid spacing
    x = linspace(0,1,N+2);
    y = linspace(0,1,N+2);
    [X, Y] = meshgrid(x,y);

    f = @(x, y) exp(-10 * ((x - 0.5).^2 + (y - 0.5).^2)); % Example source function

    e = ones(N,1);
    T = spdiags([-e 2*e -e], [-1 0 1], N, N) / h^2;
    I = speye(N);
    A = kron(I, T) + kron(T, I);
    Ax = kron(I, spdiags([-e e], [-1 1], N, N)) / (2*h);
    Ay = kron(spdiags([-e e], [-1 1], N, N), I) / (2*h);

    A = A + alpha * Ax + beta * Ay;

    b = zeros(N*N,1);
    for j = 1:N
        for i = 1:N
            idx = (j-1)*N + i;
            b(idx) = f(x(i+1), y(j+1));
        end
    end

    w0 = zeros(N*N,1);
    tol = 1e-4;

    [W, flag, relres, iter, resvec, eigest] = pcg(A, b, tol, max_iter);
end
