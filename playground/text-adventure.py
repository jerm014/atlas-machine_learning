# text_adventure.py

class Room:
    """A room in the adventure game."""

    def __init__(self, name, short_description, long_description):
        """Initialize a room with a name, short description, and long description."""
        self.name = name
        self.short_description = short_description
        self.long_description = long_description
        self.exits = {}
        self.items = []
        self.character = None
        self.visited = False  # Track if the room has been visited

    def add_exit(self, direction, room):
        """Add an exit to the room."""
        self.exits[direction] = room

    def add_item(self, item):
        """Add an item to the room."""
        self.items.append(item)

    def remove_item(self, item):
        """Remove an item from the room."""
        self.items.remove(item)

    def add_character(self, character):
        """Add a character to the room."""
        self.character = character

    def describe(self):
        """Return the appropriate description based on whether the room has been visited."""
        if not self.visited:
            self.visited = True
            return f"{self.name}\n{self.long_description}"
        else:
            return f"{self.name}\n{self.short_description}"

    def __str__(self):
        """Return a string representation of the room."""
        room_info = self.describe()
        if self.items:
            room_info += "\nYou see: " + ", ".join(self.items) + "\n"
        if self.character:
            room_info += f"You see {self.character.name} here.\n"
        return room_info


class Character:
    """A character in the adventure game."""

    def __init__(self, name, dialogue, required_item=None, reward_item=None):
        """Initialize a character with a name, dialogue, and optional required/reward items."""
        self.name = name
        self.dialogue = dialogue
        self.required_item = required_item
        self.reward_item = reward_item

    def talk(self, player):
        """Interact with the character."""
        if self.required_item and self.required_item not in player.inventory:
            print(f"{self.name}: 'You need a {self.required_item} to help me.'")
        else:
            print(f"{self.name}: '{self.dialogue}'")
            if self.reward_item:
                print(f"{self.name} gives you a {self.reward_item}!")
                player.inventory.append(self.reward_item)


class Player:
    """The player in the adventure game."""

    def __init__(self, current_room):
        """Initialize the player with a starting room."""
        self.current_room = current_room
        self.inventory = []

    def move(self, direction):
        """Move the player to a new room."""
        if direction in self.current_room.exits:
            self.current_room = self.current_room.exits[direction]
            print(f"You move to the {self.current_room.name}.")
        else:
            print("You can't go that way.")

    def take_item(self, item):
        """Take an item from the current room."""
        if item in self.current_room.items:
            self.current_room.remove_item(item)
            self.inventory.append(item)
            print(f"You take the {item}.")
        else:
            print(f"There is no {item} here.")

    def drop_item(self, item):
        """Drop an item in the current room."""
        if item in self.inventory:
            self.inventory.remove(item)
            self.current_room.add_item(item)
            print(f"You drop the {item}.")
        else:
            print(f"You don't have a {item}.")

    def show_inventory(self):
        """Show the player's inventory."""
        if self.inventory:
            print("You are carrying:")
            for item in self.inventory:
                print(f"- {item}")
        else:
            print("You are not carrying anything.")

    def examine_room(self):
        """Examine the current room, showing its long description."""
        print(f"{self.current_room.name}\n{self.current_room.long_description}")
        if self.current_room.items:
            print("You see: " + ", ".join(self.current_room.items))
        if self.current_room.character:
            print(f"You see {self.current_room.character.name} here.")


def main():
    """Main function to run the game."""

    # Create rooms with vibrant descriptions
    entrance = Room(
        name="Entrance",
        short_description="You are at the entrance of a dark cave.",
        long_description=(
            "You stand at the mouth of a dark, foreboding cave. The air is damp and cool, "
            "and the faint sound of dripping water echoes from within. The cave entrance is "
            "framed by jagged rocks, and a faint breeze brushes past you, carrying with it "
            "the scent of earth and moss. A faint glimmer of light can be seen deeper inside."
        )
    )

    hall = Room(
        name="Hall",
        short_description="You are in a long, dimly lit hall.",
        long_description=(
            "The hall stretches before you, its walls lined with ancient, crumbling tapestries. "
            "The air is thick with dust, and the faint light from a few flickering torches casts "
            "long shadows on the stone floor. The hall branches off in several directions, and "
            "the faint sound of footsteps echoes in the distance."
        )
    )

    kitchen = Room(
        name="Kitchen",
        short_description="You are in a dusty kitchen. There's a strange smell.",
        long_description=(
            "The kitchen is a mess, with pots and pans scattered across the floor and shelves. "
            "A large, rusted stove sits in the corner, and the air is filled with the faint smell "
            "of something burnt. A wooden table in the center of the room is covered in dust, "
            "and a single, chipped cup sits on its surface."
        )
    )

    library = Room(
        name="Library",
        short_description="You are in a room filled with ancient books.",
        long_description=(
            "The library is a vast, towering room filled with shelves that stretch from floor to ceiling. "
            "The shelves are crammed with ancient, leather-bound books, their spines cracked and faded. "
            "A thick layer of dust covers everything, and the air smells of old parchment. In the center "
            "of the room, a large, ornate desk sits, piled high with scrolls and tomes."
        )
    )

    dungeon = Room(
        name="Dungeon",
        short_description="You are in a dark, damp dungeon. It feels ominous.",
        long_description=(
            "The dungeon is a cold, oppressive place, with walls made of rough-hewn stone. "
            "The air is damp and heavy, and the faint sound of dripping water echoes through the darkness. "
            "Rusted iron bars line the walls, and the remains of old chains hang from the ceiling. "
            "A faint, eerie light flickers from a torch mounted on the wall, casting long shadows across the floor."
        )
    )

    treasure_room = Room(
        name="Treasure Room",
        short_description="You are in a room filled with gold and jewels!",
        long_description=(
            "The treasure room is a dazzling sight, with piles of gold coins, glittering jewels, "
            "and precious artifacts scattered across the floor. The walls are lined with shelves "
            "overflowing with treasures, and a large, ornate chest sits in the center of the room. "
            "The air is thick with the scent of wealth and power, and the faint sound of clinking coins "
            "echoes through the chamber."
        )
    )

    # Add exits
    entrance.add_exit("north", hall)
    hall.add_exit("south", entrance)
    hall.add_exit("east", kitchen)
    hall.add_exit("west", library)
    kitchen.add_exit("west", hall)
    library.add_exit("east", hall)
    library.add_exit("down", dungeon)
    dungeon.add_exit("up", library)
    dungeon.add_exit("north", treasure_room)
    treasure_room.add_exit("south", dungeon)

    # Add items
    kitchen.add_item("cup")
    library.add_item("book")
    dungeon.add_item("torch")

    # Add characters
    wizard = Character(
        name="Wizard",
        dialogue="Thank you for the book! Here's a magical cup for your troubles.",
        required_item="book",
        reward_item="magical cup"
    )
    library.add_character(wizard)

    # Initialize player
    player = Player(entrance)

    # Game loop
    while True:
        print("\n" + str(player.current_room))
        if player.current_room == treasure_room and "magical cup" in player.inventory:
            print("Congratulations! You found the treasure!")
            break

        command = input("\nWhat do you want to do? ").strip().lower()

        if command.startswith("go "):
            direction = command.split(" ")[1]
            player.move(direction)
        elif command.startswith("take "):
            item = command.split(" ")[1]
            player.take_item(item)
        elif command.startswith("drop "):
            item = command.split(" ")[1]
            player.drop_item(item)
        elif command == "inventory":
            player.show_inventory()
        elif command.startswith("talk "):
            if player.current_room.character:
                player.current_room.character.talk(player)
            else:
                print("There's no one here to talk to.")
        elif command in ["examine", "exam"]:
            player.examine_room()
        elif command == "quit":
            print("Thanks for playing!")
            break
        else:
            print("I don't understand that command.")


if __name__ == "__main__":
    main()