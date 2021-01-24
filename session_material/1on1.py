from pyschedule import Scenario, solvers, plotters, alt

# period 1 :: Wed 10:00-10:30
# period 2 :: Wed 10:30-11:00
# period 3 :: Wed 11:00-11:30
# period 4 :: Wed 11:30-12:00
# period 5 :: Fri 10:00-10:30
# period 6 :: Fri 10:30-11:00
# period 7 :: Fri 11:00-11:30
# period 8 :: Fri 11:30-12:00

S = Scenario('meetings', horizon=8)

# declare tasks
#Hibiki = S.Task('Hibiki', length=1, periods=[1])
Fred = S.Task('Fred', length=1, periods=[3, 4])
Maurice = S.Task('Maurice', length=1, periods=[3, 2])
Felix = S.Task('Felix', length=1, periods=[3, 4, 5])
Florian = S.Task('Florian', length=1, periods=[5, 6, 7])
Pablo = S.Task('Pablo', length=1, periods=[1, 2, 5, 6])
Andreas = S.Task('Andreas', length=1, periods=[7, 8])
Vince = S.Task('Vince', length=1, periods=[1, 8])
#Hong = S.Task('Hong', length=1, periods=[3,7])

# declare resources
Teacher = S.Resource('Teacher')

# declare dependencies between tasks and resources
#Hibiki += Teacher
Fred += Teacher
Maurice += Teacher
Felix += Teacher
Florian += Teacher
Pablo += Teacher
Vince += Teacher
#Hong += Teacher
Andreas += Teacher

solvers.mip.solve(S, msg=1)
print(S.solution())
