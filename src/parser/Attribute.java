package parser;

public class Attribute {
	
	private String name, value;
	
	public Attribute(String name, String value){
		this.name = name;
		this.value = value;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getValue() {
		return value;
	}

	public void setValue(String value) {
		this.value = value;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((name == null) ? 0 : name.hashCode());
		result = prime * result + ((value == null) ? 0 : value.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Attribute other = (Attribute) obj;
		if (name == null) {
			if (other.name != null)
				return false;
		} else if (!name.equals(other.name))
			return false;
		if (value == null) {
            return other.value == null;
		} else return value.equals(other.value);
    }

	public boolean equalAttribute(Attribute b){
		return this.name.equals(b.name);
	}

	public static Attribute parse(String c) {
	
		if(c.contains(";")){
			c = c.substring(0, c.length() - 1);
		}
		String[] split = c.split(":");
		if(split.length != 2){
			return null;
		}
		return new Attribute(split[0].trim(), split[1].trim());
	}

	public String toParse(int spacesLeft) {
		return ParserTools.createSpaces(spacesLeft) + name + " : " + value + ";";
	}
}
