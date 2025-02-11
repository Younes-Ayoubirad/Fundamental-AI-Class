:- dynamic(grid_size/2).
:- dynamic(agent/2).
:- dynamic(obstacle/2).

inside_grid(X, Y) :-
    grid_size(MaxX, MaxY),
    X >= 1, X =< MaxX,
    Y >= 1, Y =< MaxY.

valid_move(X, Y) :-
    inside_grid(X, Y),
    \+ obstacle(X, Y).

move((X, Y), (NX, Y)) :- NX is X + 1, valid_move(NX, Y).
move((X, Y), (NX, Y)) :- NX is X - 1, valid_move(NX, Y).
move((X, Y), (X, NY)) :- NY is Y + 1, valid_move(X, NY).
move((X, Y), (X, NY)) :- NY is Y - 1, valid_move(X, NY).

shortest_path(Start, Goal, Path, MaxDepth) :-
    bfs([[Start]], Goal, Path, MaxDepth, 0).

bfs([[Goal | Rest] | _], Goal, Path, _, _) :-
    reverse([Goal | Rest], Path).

bfs([CurrentPath | OtherPaths], Goal, Path, MaxDepth, Depth) :-
    Depth < MaxDepth,
    CurrentPath = [CurrentNode | _],
    findall([NextNode | CurrentPath],
            (move(CurrentNode, NextNode),
             \+ member(NextNode, CurrentPath)),
            NewPaths),
    append(OtherPaths, NewPaths, UpdatedPaths),
    NewDepth is Depth + 1,
    bfs(UpdatedPaths, Goal, Path, MaxDepth, NewDepth).

