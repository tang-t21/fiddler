def read_experts(filename, num_limit=56):
    experts = []
    with open(filename, "r") as f:
        for line in f:
            nums = line.split(",")
            experts.append(int(nums[0]))
    return set(experts[:num_limit])


if __name__ == "__main__":
    categories = [
        "writing",
        "roleplay",
        "reasoning",
        "math",
        "coding",
        "extraction",
        "stem",
        "humanities",
    ]
    experts = []
    for i in range(len(categories)):
        experts.append(read_experts(f"{categories[i]}.txt"))
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            print(
                f"Intersection between {categories[i]} and {categories[j]}: {len(experts[i].intersection(experts[j]))}"
            )
