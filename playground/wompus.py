#!/usr/bin/env python3
import random

class WumpusGame:
    def __init__(self):
        # Cave system - 20 rooms with 3 tunnels each
        self.cave = {
            1: [2, 5, 8], 2: [1, 3, 10], 3: [2, 4, 12], 4: [3, 5, 14], 5: [1, 4, 6],
            6: [5, 7, 15], 7: [6, 8, 17], 8: [1, 7, 9], 9: [8, 10, 18], 10: [2, 9, 11],
            11: [10, 12, 19], 12: [3, 11, 13], 13: [12, 14, 20], 14: [4, 13, 15], 15: [6, 14, 16],
            16: [15, 17, 20], 17: [7, 16, 18], 18: [9, 17, 19], 19: [11, 18, 20], 20: [13, 16, 19]
        }
        self.arrows = 5
        self.reset_game()

    def reset_game(self):
        # Place player, wumpus, pits, and bats
        locations = random.sample(range(1, 21), 6)
        self.player_loc = locations[0]
        self.wumpus_loc = locations[1]
        self.pit_locs = locations[2:4]
        self.bat_locs = locations[4:6]
        self.initial_locations = locations[:]

    def display_instructions(self):
        instructions = """
WELCOME TO 'HUNT THE WUMPUS'
  The Wumpus lives in a cave of 20 rooms. Each room
has 3 tunnels leading to other rooms. (Look at a
dodecahedron to see how this works-if you don't know
what a dodecahedron is, ask someone)

     HAZARDS:
 BOTTOMLESS PITS - Two rooms have bottomless pits in them
     If you go there, you fall into the pit (& lose!)
 SUPER BATS - Two other rooms have super bats. If you
     go there, a bat grabs you and takes you to some other
     room at random. (Which might be troublesome)

     WUMPUS:
 The Wumpus is not bothered by hazards (he has sucker
 feet and is too big for a bat to lift). Usually
 he is asleep. Two things wake him up: your entering
 his room or your shooting an arrow.
     If the Wumpus wakes, he moves (P=.75) one room
 or stays still (P=.25). After that, if he is where you
 are, he eats you up (& you lose!)

     YOU:
 Each turn you may move or shoot a crooked arrow
   MOVING: You can go one room (through one tunnel)
   ARROWS: You have 5 arrows. You lose when you run out.
   Each arrow can go from 1 to 5 rooms. You aim by telling
   the computer the room#s you want the arrow to go to.
   If the arrow can't go that way (i.e., no tunnel) it moves
   at random to the next room.
     If the arrow hits the Wumpus, you win.
     If the arrow hits you, you lose.

    WARNINGS:
     When you are one room away from Wumpus or hazard,
    the computer says:
 WUMPUS:  'I smell a Wumpus'
 BAT   :  'Bats nearby'
 PIT   :  'I feel a draft'
"""
        print(instructions)
        input("Press Enter to continue...")

    def check_hazards(self):
        """Print warnings about nearby hazards"""
        for tunnel in self.cave[self.player_loc]:
            if tunnel == self.wumpus_loc:
                print("I smell a Wumpus!")
            if tunnel in self.pit_locs:
                print("I feel a draft!")
            if tunnel in self.bat_locs:
                print("Bats nearby!")

    def move_wumpus(self):
        """Move wumpus with 75% probability"""
        if random.random() < 0.75:
            self.wumpus_loc = random.choice(self.cave[self.wumpus_loc])
        return self.wumpus_loc == self.player_loc

    def shoot_arrow(self):
        """Handle arrow shooting logic"""
        # Get number of rooms
        while True:
            try:
                num_rooms = int(input("Number of rooms (1-5)? "))
                if 1 <= num_rooms <= 5:
                    break
                print("Please enter a number between 1 and 5.")
            except ValueError:
                print("Please enter a valid number.")

        # Get room numbers
        path = []
        current_room = self.player_loc
        for i in range(num_rooms):
            while True:
                try:
                    room = int(input(f"Room #{i+1}? "))
                    if len(path) >= 2 and room == path[-2]:
                        print("Arrows aren't that crooked - try another room")
                        continue
                    path.append(room)
                    break
                except ValueError:
                    print("Please enter a valid room number.")

        # Track arrow
        for room in path:
            if current_room == self.player_loc and room == self.wumpus_loc:
                return "WIN"
            if room in self.cave[current_room]:
                current_room = room
            else:
                current_room = random.choice(self.cave[current_room])
            
            if current_room == self.wumpus_loc:
                return "WIN"
            elif current_room == self.player_loc:
                return "SELF"

        self.arrows -= 1
        if self.arrows == 0:
            return "NO_ARROWS"
        
        print("Missed!")
        if self.move_wumpus():
            return "EATEN"
        return "CONTINUE"

    def move_player(self):
        """Handle player movement"""
        while True:
            try:
                new_loc = int(input("Where to? "))
                if new_loc in self.cave[self.player_loc]:
                    break
                print("Not possible - ", end="")
            except ValueError:
                print("Please enter a valid room number. ", end="")

        self.player_loc = new_loc

        # Check for wumpus
        if self.player_loc == self.wumpus_loc:
            print("... Oops! Bumped a Wumpus!")
            if self.move_wumpus():
                return "EATEN"

        # Check for pits
        if self.player_loc in self.pit_locs:
            print("YYYIIIIEEEE . . . Fell in pit")
            return "PIT"

        # Check for bats
        if self.player_loc in self.bat_locs:
            print("ZAP--Super bat snatch! Elsewhereville for you!")
            self.player_loc = random.randint(1, 20)
            return self.move_player()  # Recursive call in case we land in another hazard

        return "CONTINUE"

    def play(self):
        """Main game loop"""
        print("\nHUNT THE WUMPUS")
        
        while True:
            print(f"\nYou are in room {self.player_loc}")
            print(f"Tunnels lead to: {self.cave[self.player_loc]}")
            self.check_hazards()

            while True:
                action = input("Shoot or Move (S-M)? ").upper()
                if action in ['S', 'M']:
                    break
                print("Please enter 'S' or 'M'")

            if action == 'S':
                result = self.shoot_arrow()
            else:
                result = self.move_player()

            if result == "WIN":
                print("AHA! You got the Wumpus!")
                return True
            elif result == "EATEN":
                print("TSK TSK TSK - Wumpus got you!")
                return False
            elif result == "PIT":
                return False
            elif result == "SELF":
                print("Ouch! Arrow got you!")
                return False
            elif result == "NO_ARROWS":
                print("Out of arrows!")
                return False

def main():
    game = WumpusGame()
    
    print("Instructions (Y-N)?", end=" ")
    if input().upper().startswith('Y'):
        game.display_instructions()
    
    print("\nATTENTION ALL WUMPUS LOVERS!!!")
    print("THERE ARE NOW 3 ADDITIONS TO THE WUMPUS FAMILY")
    print("OF PROGRAMS.")
    print("\nWUMP2:  SOME DIFFERENT CAVE ARRANGEMENTS")
    print("WUMP3:  DIFFERENT HAZARDS")
    print("WUMP4:  HIDE-N-SEEK")
    
    while True:
        won = game.play()
        if won:
            print("HEE HEE HEE - THE WUMPUS'LL GETCHA NEXT TIME!!")
        else:
            print("HA HA HA - YOU LOSE!")
            
        print("\nSame setup (Y-N)?", end=" ")
        if input().upper().startswith('Y'):
            game.player_loc = game.initial_locations[0]
            game.wumpus_loc = game.initial_locations[1]
            game.pit_locs = game.initial_locations[2:4]
            game.bat_locs = game.initial_locations[4:6]
            game.arrows = 5
        else:
            game.reset_game()
            game.arrows = 5

if __name__ == "__main__":
    main()
